import os, json, sqlite3, datetime, cv2, requests
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from measurements import analyze_image
from roboflow_api import classify_breed
from breed_data import BREED_DATA

load_dotenv()

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = 'records.db'
JSONL_PATH = 'records.jsonl'
ALLOWED_EXT = {'png','jpg','jpeg'}

BPA_ENDPOINT = os.getenv('BPA_ENDPOINT')
BPA_API_KEY = os.getenv('BPA_API_KEY')

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET', 'devkey')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize DB
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    filename TEXT,
                    data JSON);''')
    conn.commit()
    conn.close()
init_db()

def allowed_file(fname): 
    return '.' in fname and fname.rsplit('.',1)[1].lower() in ALLOWED_EXT

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '': 
            flash("No file selected")
            return redirect(request.url)
        if not allowed_file(file.filename): 
            flash("Invalid file type")
            return redirect(request.url)

        fname = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        file.save(path)

        # Analyze physical traits
        try:
            result = analyze_image(path)
        except Exception as e:
            flash("Analysis failed: "+str(e))
            return redirect(request.url)

        annotated = result.pop('annotated_image')

        # Breed classification from Roboflow
        breed_result = classify_breed(path)

        # Draw bbox only if valid
        bbox = breed_result.get("bbox")
        if bbox:
            x = bbox.get("x")
            y = bbox.get("y")
            w = bbox.get("width")
            h = bbox.get("height")
            if None not in (x, y, w, h):
                pt1 = (int(x - w/2), int(y - h/2))
                pt2 = (int(x + w/2), int(y + h/2))
                cv2.rectangle(annotated, pt1, pt2, (0,0,255), 2)
                cv2.putText(annotated, breed_result.get("breed","Unknown"), pt1,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        ann_name = f"annotated_{fname}"
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], ann_name), annotated)

        # Create record
        timestamp = datetime.datetime.now().astimezone().isoformat()
        record = {
            "filename": fname,
            "timestamp": timestamp,
            "measurements": result,
            "classification": {
                "breed": breed_result.get("breed", "Unknown"),
                "score": breed_result.get("score", 0),
                "breed_info": BREED_DATA.get(breed_result.get("breed", ""), {})
            }
        }

        # Save to JSONL
        with open(JSONL_PATH, 'a') as f:
            f.write(json.dumps(record)+"\n")

        # Save to SQLite DB
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute('INSERT INTO records (timestamp, filename, data) VALUES (?,?,?)',
                    (timestamp, fname, json.dumps(record)))
        conn.commit()
        conn.close()

        # Send to BPA if configured
        sent_to_bpa = False
        if BPA_ENDPOINT:
            try:
                headers = {'Content-Type':'application/json'}
                if BPA_API_KEY:
                    headers['Authorization'] = f'Bearer {BPA_API_KEY}'
                r = requests.post(BPA_ENDPOINT, json=record, headers=headers, timeout=10)
                sent_to_bpa = r.ok
            except Exception as e:
                print("BPA send failed:", e)
                sent_to_bpa = False

        return render_template('index.html', result=record,
                               annotated_url=url_for('uploaded_file', filename=ann_name),
                               sent_to_bpa=sent_to_bpa)
    return render_template('index.html', result=None)

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000)

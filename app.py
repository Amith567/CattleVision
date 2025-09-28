import os
import cv2
import numpy as np
import math
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# Import breed dictionary
from breed_data import BREED_DATA  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ----------------- Image Analysis -----------------
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Cannot load image: " + path)
    return img

def get_largest_contour(gray):
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        th = cv2.bitwise_not(th)
        cnts, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def pca_orientation(contour):
    data = contour.reshape(-1,2).astype(np.float64)
    mean = data.mean(axis=0)
    cov = np.cov((data-mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argmax(eigvals)
    major = eigvecs[:, idx]
    angle = math.atan2(major[1], major[0])
    return angle, mean

def rotate_points(pts, angle, center):
    ca = math.cos(-angle)
    sa = math.sin(-angle)
    cx, cy = center
    pts_rot = []
    for (x,y) in pts:
        xr = ca*(x-cx) - sa*(y-cy) + cx
        yr = sa*(x-cx) + ca*(y-cy) + cy
        pts_rot.append([xr, yr])
    return np.array(pts_rot)

def analyze_image(image_path, reference_pixel_length=None, reference_real_length_cm=None):
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contour = get_largest_contour(gray)
    if contour is None or cv2.contourArea(contour) < 5000:
        raise ValueError("No valid contour detected")

    angle, center = pca_orientation(contour)
    pts = contour.reshape(-1,2)
    pts_rot = rotate_points(pts, angle, center)

    xs, ys = pts_rot[:,0], pts_rot[:,1]
    min_x, max_x, min_y, max_y = xs.min(), xs.max(), ys.min(), ys.max()

    # Traits in pixels
    body_length_px = float(max_x - min_x)
    height_withers_px = float(max_y - min_y)
    chest_width_px = float(height_withers_px * 0.4)  # heuristic
    rump_angle_deg = float(round(math.degrees(angle), 2))

    # Convert to cm if reference provided
    if reference_pixel_length and reference_real_length_cm:
        pixels_per_cm = reference_pixel_length / reference_real_length_cm
        body_length_cm = body_length_px / pixels_per_cm
        height_withers_cm = height_withers_px / pixels_per_cm
        chest_width_cm = chest_width_px / pixels_per_cm
    else:
        body_length_cm = height_withers_cm = chest_width_cm = None

    annotated = img.copy()
    cv2.drawContours(annotated, [contour], -1, (0,255,0), 2)
    cv2.putText(annotated, f"Length: {body_length_px:.1f}px", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    
    return {
        "body_length_px": body_length_px,
        "height_withers_px": height_withers_px,
        "chest_width_px": chest_width_px,
        "rump_angle_deg": rump_angle_deg,
        "body_length_cm": body_length_cm,
        "height_withers_cm": height_withers_cm,
        "chest_width_cm": chest_width_cm,
        "annotated_image": annotated
    }

# ----------------- Flask Routes -----------------
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    result_json = None
    uploaded_image_url = None

    if request.method == "POST":
        if "image" in request.files:
            image_file = request.files["image"]
            if image_file.filename != "":
                filename = secure_filename(image_file.filename.lower())
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(image_path)
                uploaded_image_url = f"/uploads/{filename}"

                # Optional reference object
                ref_pixel_length = request.form.get("ref_pixel_length", type=float)
                ref_real_length_cm = request.form.get("ref_real_length_cm", type=float)

                # Run analysis
                analysis = analyze_image(
                    image_path,
                    reference_pixel_length=ref_pixel_length,
                    reference_real_length_cm=ref_real_length_cm
                )

                # Save annotated image
                annotated_file = os.path.join(app.config['UPLOAD_FOLDER'], "annotated_"+filename)
                cv2.imwrite(annotated_file, analysis["annotated_image"])
                annotated_url = f"/uploads/annotated_{filename}"

                # For now: hardcoded breed (later can replace with AI detection)
                detected_breed = "Murrah Buffalo"
                breed_info = BREED_DATA.get(detected_breed, {})

                result_json = {
                    "filename": filename,
                    "measurements": {
                        "body_length_px": analysis["body_length_px"],
                        "height_withers_px": analysis["height_withers_px"],
                        "chest_width_px": analysis["chest_width_px"],
                        "rump_angle_deg": analysis["rump_angle_deg"],
                        "body_length_cm": analysis["body_length_cm"],
                        "height_withers_cm": analysis["height_withers_cm"],
                        "chest_width_cm": analysis["chest_width_cm"]
                    },
                    "annotated_url": annotated_url,
                    "breed": {
                        "name": detected_breed,
                        "info": breed_info
                    }
                }

    return render_template(
        "index.html",
        result=result_json,
        uploaded_image_url=uploaded_image_url
    )

if __name__ == "__main__":
    app.run(debug=True)

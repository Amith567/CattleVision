import requests, os
from dotenv import load_dotenv

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL = os.getenv("ROBOFLOW_MODEL")   # e.g. "cattle-breeds/1"
ROBOFLOW_ENDPOINT = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}"

def classify_breed(image_path):
    with open(image_path, "rb") as f:
        resp = requests.post(
            ROBOFLOW_ENDPOINT,
            params={"api_key": ROBOFLOW_API_KEY},
            files={"file": f}
        )
    if resp.status_code != 200:
        return {"breed": "Unknown", "score": 0, "error": resp.text}

    data = resp.json()

    if "predictions" in data and data["predictions"]:
        best = max(data["predictions"], key=lambda x: x.get("confidence", 0))
        return {
            "breed": best.get("class", "Unknown"),
            "score": round(best.get("confidence", 0) * 100, 2),
            "bbox": {
                "x": best.get("x"), "y": best.get("y"),
                "width": best.get("width"), "height": best.get("height")
            }
        }
    return {"breed": "Unknown", "score": 0}

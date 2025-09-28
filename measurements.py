import cv2
import numpy as np
import math

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
    ca = math.cos(-angle); sa = math.sin(-angle)
    cx, cy = center
    pts_rot = []
    for (x,y) in pts:
        xr = ca*(x-cx) - sa*(y-cy) + cx
        yr = sa*(x-cx) + ca*(y-cy) + cy
        pts_rot.append([xr, yr])
    return np.array(pts_rot)

def analyze_image(image_path, reference_pixel_length=None, reference_real_length_cm=None):
    """
    Analyze animal image and return measurements.
    
    reference_pixel_length: pixels of known reference object in image
    reference_real_length_cm: real length of reference object in cm
    """
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

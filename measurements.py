# measurements.py
import cv2
import numpy as np
import math

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Cannot load image: " + path)
    return img

def get_largest_contour_from_thresh(gray):
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        th = cv2.bitwise_not(th)
        cnts, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, th
    return max(cnts, key=cv2.contourArea), th

def pca_orientation(contour):
    data = contour.reshape(-1,2).astype(np.float64)
    mean = data.mean(axis=0)
    cov = np.cov((data-mean).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argmax(eigvals)
    major = eigvecs[:, idx]
    angle = math.atan2(major[1], major[0])  # radians
    return angle, mean

def rotate_points(pts, theta, center):
    """
    Rotate points by theta (radians) counter-clockwise around center (cx,cy).
    Positive theta rotates CCW.
    """
    cx, cy = center
    c = math.cos(theta); s = math.sin(theta)
    pts_rot = []
    for (x,y) in pts:
        xr = c*(x-cx) - s*(y-cy) + cx
        yr = s*(x-cx) + c*(y-cy) + cy
        pts_rot.append([xr, yr])
    return np.array(pts_rot)

def rotate_point(pt, theta, center):
    cx, cy = center
    c = math.cos(theta); s = math.sin(theta)
    x, y = pt
    xr = c*(x-cx) - s*(y-cy) + cx
    yr = s*(x-cx) + c*(y-cy) + cy
    return (xr, yr)

def fill_holes(mask):
    # mask: uint8 0/255
    h, w = mask.shape[:2]
    im_flood = mask.copy()
    mask_ff = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_flood, mask_ff, (0,0), 255)
    im_flood_inv = cv2.bitwise_not(im_flood)
    filled = mask | im_flood_inv
    return filled

def keep_largest_component(bin_mask):
    cnts, _ = cv2.findContours(bin_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bin_mask
    largest = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(bin_mask)
    cv2.drawContours(out, [largest], -1, 255, -1)
    return out

def segment_with_grabcut(img, fallback_thresh=True, iter_gc=5):
    """
    Try to create a clean foreground mask using:
      1) a threshold-based initial bounding box
      2) cv2.grabCut with that bounding box
    Returns a binary mask (uint8 0/255)
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) find a coarse largest contour from Otsu threshold
    contour, th = get_largest_contour_from_thresh(gray)
    if contour is None:
        # fallback: simple rectangle around center
        rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))
    else:
        x,y,ww,hh = cv2.boundingRect(contour)
        padx = int(ww * 0.08) + 10
        pady = int(hh * 0.08) + 10
        x0 = max(0, x - padx); y0 = max(0, y - pady)
        x1 = min(w, x + ww + padx); y1 = min(h, y + hh + pady)
        rect = (x0, y0, x1 - x0, y1 - y0)

    # GrabCut
    mask_gc = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(img, mask_gc, rect, bgdModel, fgdModel, iter_gc, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask_gc==2)|(mask_gc==0), 0, 1).astype('uint8')*255
    except Exception:
        # if grabcut fails, fallback to threshold mask
        mask2 = th.copy()

    # Clean mask: morphological close -> fill holes -> keep largest component
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask2 = fill_holes(mask2)
    mask2 = keep_largest_component(mask2)
    # Smooth edges
    mask2 = cv2.medianBlur(mask2, 5)
    return mask2

def contour_from_mask(mask):
    cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)

def analyze_image(image_path, reference_pixel_length=None, reference_real_length_cm=None):
    """
    Improved analyze_image with GrabCut-based segmentation + refined measurements.
    Returns same structure as before with 'annotated_image' as a BGR numpy image.
    """
    img = load_image(image_path)
    h, w = img.shape[:2]

    # 1) Segment
    mask = segment_with_grabcut(img)
    if mask is None or cv2.countNonZero(mask) < 5000:
        # fallback: try simple threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Keep largest component again (safety)
    mask = keep_largest_component(mask)
    # produce contour from mask (full)
    contour = contour_from_mask(mask)
    if contour is None or cv2.contourArea(contour) < 5000:
        raise ValueError("No valid contour detected")

    # 2) PCA orientation on full contour (so we align animal horizontally)
    angle, center = pca_orientation(contour)
    # rotate contour points by -angle (so major axis -> x-axis)
    pts = contour.reshape(-1,2)
    pts_rot = rotate_points(pts, -angle, center)

    xs, ys = pts_rot[:,0], pts_rot[:,1]
    min_x, max_x, min_y, max_y = xs.min(), xs.max(), ys.min(), ys.max()
    body_length_px_raw = float(max_x - min_x)
    height_px_raw = float(max_y - min_y)

    # 3) Erode the mask to remove legs/tail (thin appendages) -> get a cleaner body shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    eroded = cv2.erode(mask, kernel, iterations=6)   # adjust iterations for stronger erosion
    eroded = keep_largest_component(eroded)
    contour_eroded = contour_from_mask(eroded)
    if contour_eroded is None or cv2.contourArea(contour_eroded) < 1000:
        # fallback: use original contour (no erosion)
        contour_eroded = contour
        eroded = mask.copy()

    pts_e = contour_eroded.reshape(-1,2)
    pts_e_rot = rotate_points(pts_e, -angle, center)
    xs_e, ys_e = pts_e_rot[:,0], pts_e_rot[:,1]
    min_x_e, max_x_e, min_y_e, max_y_e = xs_e.min(), xs_e.max(), ys_e.min(), ys_e.max()

    # Body length using eroded (less influenced by tail/legs)
    body_length_px = float(max_x_e - min_x_e)
    height_withers_px = float(max_y_e - min_y_e)

    # Chest width: measure at a row ~35% from top of eroded body
    chest_y = int(min_y_e + 0.35 * (max_y_e - min_y_e))
    # find left/right boundary at chest_y on eroded mask (in rotated coords)
    # To do this, create rotated mask and read the row
    # Create rotated mask by rotating mask image
    # We'll create a blank mask image and rotate using warpAffine

    # create mask image from eroded contour in original space, then rotate
    eroded_img = np.zeros((h,w), np.uint8)
    cv2.drawContours(eroded_img, [contour_eroded], -1, 255, -1)
    # rotate the eroded_img around center by -angle
    M = cv2.getRotationMatrix2D(tuple(center), -math.degrees(angle), 1.0)
    rotated_eroded = cv2.warpAffine(eroded_img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    chest_y_int = int(round(chest_y))
    chest_y_int = max(0, min(h-1, chest_y_int))
    row = rotated_eroded[chest_y_int]
    xs_row = np.where(row > 0)[0]
    if xs_row.size >= 2:
        chest_left_x = float(xs_row.min())
        chest_right_x = float(xs_row.max())
        chest_width_px = chest_right_x - chest_left_x
    else:
        # fallback heuristic
        chest_width_px = 0.4 * height_withers_px

    # Rump angle: estimate slope of the upper-back near the rump (right end)
    # Use rotated_eroded contour points (rotated coords)
    # pick points within x > max_x_e - 0.15*body_length
    cutoff = max_x_e - 0.15 * (max_x_e - min_x_e)
    mask_pts = pts_e_rot
    right_region_pts = mask_pts[mask_pts[:,0] >= cutoff]
    # keep the top-most of those (upper-back): choose points with y < min_y_e + 0.5*height
    upper_back_pts = right_region_pts[right_region_pts[:,1] <= (min_y_e + 0.5*(max_y_e - min_y_e))]
    if len(upper_back_pts) >= 3:
        xs_fit = upper_back_pts[:,0]
        ys_fit = upper_back_pts[:,1]
        # fit linear model y = m*x + c
        m, c = np.polyfit(xs_fit, ys_fit, 1)
        rump_angle_rad = math.atan(m)
        rump_angle_deg = float(round(math.degrees(rump_angle_rad), 2))
    else:
        # fallback to PCA angle (converted to degrees)
        rump_angle_deg = float(round(math.degrees(angle), 2))

    # Convert to cm if reference provided
    if reference_pixel_length and reference_real_length_cm:
        pixels_per_cm = reference_pixel_length / reference_real_length_cm
        body_length_cm = body_length_px / pixels_per_cm
        height_withers_cm = height_withers_px / pixels_per_cm
        chest_width_cm = chest_width_px / pixels_per_cm
    else:
        body_length_cm = height_withers_cm = chest_width_cm = None

    # Annotate original image: draw contours (original mask), and draw the measurement lines
    annotated = img.copy()
    cv2.drawContours(annotated, [contour], -1, (0,200,0), 2)  # main contour

    # compute endpoints in rotated coords and convert back to original coords to draw lines
    left_rot = (min_x_e, (min_y_e + max_y_e)/2)
    right_rot = (max_x_e, (min_y_e + max_y_e)/2)
    top_rot = ((min_x_e + max_x_e)/2, min_y_e)
    bottom_rot = ((min_x_e + max_x_e)/2, max_y_e)

    # rotate back by +angle to original image space
    left_orig = tuple(map(int, rotate_point(left_rot, angle, center)))
    right_orig = tuple(map(int, rotate_point(right_rot, angle, center)))
    top_orig = tuple(map(int, rotate_point(top_rot, angle, center)))
    bottom_orig = tuple(map(int, rotate_point(bottom_rot, angle, center)))

    # Body length line
    cv2.line(annotated, left_orig, right_orig, (0,255,0), 3)
    cv2.putText(annotated, f"Body: {body_length_px:.1f}px", (left_orig[0], left_orig[1]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Height line
    cv2.line(annotated, top_orig, bottom_orig, (0,255,0), 3)
    cv2.putText(annotated, f"Height: {height_withers_px:.1f}px", (top_orig[0]+8, top_orig[1]+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Chest width: endpoints in rotated coords then back
    chest_left_rot = (chest_left_x, chest_y_int)
    chest_right_rot = (chest_right_x if xs_row.size>=2 else chest_left_x + chest_width_px, chest_y_int)
    chest_left_orig = tuple(map(int, rotate_point(chest_left_rot, angle, center)))
    chest_right_orig = tuple(map(int, rotate_point(chest_right_rot, angle, center)))

    cv2.line(annotated, chest_left_orig, chest_right_orig, (0,255,0), 3)
    cv2.putText(annotated, f"Chest: {chest_width_px:.1f}px", (chest_left_orig[0], chest_left_orig[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Optionally draw a small marker for withers (approx: top_orig)
    cv2.circle(annotated, (int(top_orig[0]), int(top_orig[1])), 5, (0,120,255), -1)

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

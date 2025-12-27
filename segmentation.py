import cv2
import numpy as np


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred


def segment_organs(image):
    gray, blurred = preprocess(image)

    _, organ_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((5, 5), np.uint8)
    organ_mask = cv2.morphologyEx(organ_mask, cv2.MORPH_CLOSE, kernel)
    organ_mask = cv2.morphologyEx(organ_mask, cv2.MORPH_OPEN, kernel)

    return organ_mask


def segment_tumor_option1(image):
    """
    Option 1: Classical tumor candidate segmentation
    """
    gray, _ = preprocess(image)
    organ_mask = segment_organs(image)

    # Work only inside organ
    roi = cv2.bitwise_and(gray, gray, mask=organ_mask)

    # ---------- REGION GROWING (via threshold range) ----------
    mean_val = np.mean(roi[organ_mask > 0])
    std_val = np.std(roi[organ_mask > 0])

    lower = int(max(mean_val - 0.5 * std_val, 0))
    upper = int(min(mean_val + 0.5 * std_val, 255))

    candidate_mask = cv2.inRange(roi, lower, upper)

    # ---------- MORPHOLOGY ----------
    kernel = np.ones((3, 3), np.uint8)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)

    # ---------- SHAPE FILTERING ----------
    final_mask = np.zeros_like(candidate_mask)
    contours, _ = cv2.findContours(
        candidate_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > 8000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = bw / bh if bh != 0 else 0

        # Reject edge-touching regions
        if x == 0 or y == 0 or (x + bw) >= w or (y + bh) >= h:
            continue

        # Tumor-like constraints
        if circularity > 0.3 and solidity > 0.6 and 0.5 < aspect_ratio < 2.0:
            cv2.drawContours(final_mask, [cnt], -1, 255, -1)

    return final_mask

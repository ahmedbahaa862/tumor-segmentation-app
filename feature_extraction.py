import cv2
import pandas as pd
import numpy as np


def extract_tumor_features(mask, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    features = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)

        if area > 0:
            perimeter = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h if h != 0 else 0

            tumor_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(tumor_mask, [cnt], -1, 255, -1)
            mean_intensity = cv2.mean(gray, mask=tumor_mask)[0]

            features.append({
                "Tumor ID": i + 1,
                "Area (pixels)": round(area, 2),
                "Perimeter": round(perimeter, 2),
                "Width": w,
                "Height": h,
                "Aspect Ratio": round(aspect_ratio, 3),
                "Mean Intensity": round(mean_intensity, 2)
            })

    return pd.DataFrame(features)

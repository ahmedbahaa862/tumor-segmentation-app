import cv2
import numpy as np
import pydicom

def read_image(uploaded_file):
    """
    Read JPG/PNG or DICOM files
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".dcm"):
        dicom = pydicom.dcmread(uploaded_file)
        image = dicom.pixel_array.astype(np.float32)

        # Normalize
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return image


def overlay_mask(image, mask):
    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)
    return overlay

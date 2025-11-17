import cv2
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch

# Initialize MTCNN
mtcnn = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')

def detect_and_crop_face(original_img_bgr, degraded_img_bgr):
    """
    Detects face in the original image using MTCNN, draws a bounding box,
    and uses that box to crop the degraded image.

    Args:
        original_img_bgr (np.ndarray): Original image in BGR format.
        degraded_img_bgr (np.ndarray): Degraded image in BGR format.

    Returns:
        tuple: (original_with_bbox, cropped_degraded)
    """
    img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Detect face
    boxes, _ = mtcnn.detect(img_pil)

    if boxes is None:
        print("No face detected.")
        return original_img_bgr, None

    x1, y1, x2, y2 = map(int, boxes[0])

    # Draw bounding box on original image
    original_with_bbox = original_img_bgr.copy()
    cv2.rectangle(original_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop from degraded image using bounding box
    h, w = degraded_img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cropped_face = degraded_img_bgr[y1:y2, x1:x2]

    return original_with_bbox, cropped_face

import torch
import numpy as np
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
from numpy.linalg import norm


# Initialize MTCNN once (so it doesn't re-load every time you call it)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn_detector = MTCNN(keep_all=True, device=device)


def select_main_face(boxes, probs, img_size, min_conf=0.9):
    """
    Select best face based on confidence, area, and distance from center.
    """
    if boxes is None or probs is None:
        return None, None

    img_center = np.array([img_size[0] / 2, img_size[1] / 2])
    best_score = -np.inf
    best_idx = None

    for i, (box, prob) in enumerate(zip(boxes, probs)):
        if prob is None or prob < min_conf:
            continue
        x0, y0, x1, y1 = box
        area = (x1 - x0) * (y1 - y0)
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        dist = norm(center - img_center)
        score = area - 2.0 * dist  # Weighted scoring

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is not None:
        return boxes[best_idx], probs[best_idx]
    return None, None


def detect_main_face(pil_img, min_conf=0.9, return_cropped=True):
    """
    Detect and return best face with bounding box drawn.

    Args:
        pil_img (PIL.Image): Input image.
        min_conf (float): Minimum face confidence.
        return_cropped (bool): Whether to return cropped face.

    Returns:
        img_with_box (PIL.Image): Image with main face box drawn.
        cropped_face (PIL.Image or None): Cropped face (or None).
        box (list): Bounding box.
        prob (float): Confidence score.
    """
    img_copy = pil_img.copy()
    boxes, probs = mtcnn_detector.detect(pil_img)

    box, prob = select_main_face(boxes, probs, pil_img.size, min_conf)

    draw = ImageDraw.Draw(img_copy)
    cropped = None

    if box is not None:
        x0, y0, x1, y1 = [int(c) for c in box]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, y0), f"{prob:.2f}", fill="red")

        if return_cropped:
            cropped = pil_img.crop((x0, y0, x1, y1))

    return img_copy, cropped, box, prob


def detect_faces(pil_img, min_conf=0.9, return_all_faces=False, draw_boxes=True):
    """
    Detect faces from an image.

    Args:
        pil_img (PIL.Image): Input image.
        min_conf (float): Minimum face confidence.
        return_all_faces (bool): If True, return all detected faces.
        draw_boxes (bool): If True, draw bounding boxes on the image.

    Returns:
        img_with_boxes (PIL.Image): Image with bounding boxes drawn.
        cropped_faces (List[PIL.Image]): List of cropped faces.
        boxes (List[List]): List of bounding boxes.
        probs (List[float]): List of confidence scores.
    """
    img_copy = pil_img.copy()
    boxes, probs = mtcnn_detector.detect(pil_img)

    if boxes is None or probs is None:
        return img_copy, [], [], []

    cropped_faces = []
    final_boxes = []
    final_probs = []

    draw = ImageDraw.Draw(img_copy)

    for box, prob in zip(boxes, probs):
        if prob is not None and prob >= min_conf:
            x0, y0, x1, y1 = [int(c) for c in box]
            if draw_boxes:
                draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                draw.text((x0, y0), f"{prob:.2f}", fill="red")
            cropped_faces.append(pil_img.crop((x0, y0, x1, y1)))
            final_boxes.append(box)
            final_probs.append(prob)

    # If return_all_faces is False, return only the main face
    if not return_all_faces and cropped_faces:
        best_box, best_prob = select_main_face(final_boxes, final_probs, pil_img.size, min_conf)
        idx = final_boxes.index(best_box)
        return img_copy, [cropped_faces[idx]], [best_box], [best_prob]

    return img_copy, cropped_faces, final_boxes, final_probs


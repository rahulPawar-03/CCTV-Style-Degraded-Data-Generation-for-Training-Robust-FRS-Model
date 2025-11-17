import cv2
import numpy as np
import os
import random
from PIL import Image


def apply_group_A(img):
    """Applies degradations related to image acquisition and sensor defects (extremely low intensity)."""
    degraded_img = img.copy()

    # 1. Very subtle Color Shift (LAB space)
    lab = cv2.cvtColor(degraded_img, cv2.COLOR_BGR2LAB)
    lab = lab.astype(np.float32)
    lab[:, :, 1] = lab[:, :, 1] * 1.05
    lab[:, :, 2] = lab[:, :, 2] * 0.95
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    degraded_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Almost no Contrast and Brightness Changes
    # This block is commented out to eliminate the dimming effect.
    # degraded_img = cv2.convertScaleAbs(degraded_img, alpha=0.9, beta=0)
    # degraded_img = cv2.convertScaleAbs(degraded_img, alpha=1.1, beta=10)

    # 3. Vignetting is commented out to prevent dimming.

    # 4. Reduced Color Channel Shift
    b, g, r = cv2.split(degraded_img)
    r_shifted = np.zeros_like(r)
    shift_amount = 1
    r_shifted[:, shift_amount:] = r[:, :-shift_amount]
    b_shifted = np.zeros_like(b)
    b_shifted[:, :-shift_amount] = b[:, shift_amount:]
    degraded_img = cv2.merge([b_shifted, g, r_shifted])

    return degraded_img

def apply_group_B(img):
    """Applies degradations related to transmission and compression artifacts (heavy effect)."""
    degraded_img = img.copy()
    degraded_img = cv2.resize(degraded_img, (128, 128), interpolation=cv2.INTER_NEAREST)
    degraded_img = cv2.resize(degraded_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 25]
    _, encimg = cv2.imencode('.jpg', degraded_img, encode_param)
    degraded_img = cv2.imdecode(encimg, 1)
    rows, cols = degraded_img.shape[:2]
    block_size = 32
    loss_probability = 0.001
    for y in range(0, rows, block_size):
        for x in range(0, cols, block_size):
            if random.random() < loss_probability:
                degraded_img[y:y+block_size, x:x+block_size] = 0
    degraded_img = degraded_img.astype(np.float32)
    step = 256 // 16
    degraded_img = (np.round(degraded_img / step) * step)
    degraded_img = np.clip(degraded_img, 0, 255).astype(np.uint8)
    return degraded_img

def apply_group_C(img):
    """Applies a single random blurring effect out of three with reduced blur."""
    degraded_img = img.copy()
    def gaussian_blur(image):
        return cv2.GaussianBlur(image, (5, 5), sigmaX=1)
    def motion_blur(image):
        size = 7
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[int((size-1)/2), :] = np.ones(size, dtype=np.float32)
        kernel /= size
        return cv2.filter2D(image, -1, kernel)
    def lens_blur(image):
        rows, cols = image.shape[:2]
        mask = np.zeros_like(image)
        cv2.circle(mask, (cols//2, rows//2), min(rows, cols)//2 - 10, (255, 255, 255), -1)
        blurred = cv2.GaussianBlur(image.copy(), (5, 5), 0)
        mask_bool = mask == 255
        return np.where(mask_bool, image, blurred)
    blur_effects = [gaussian_blur, motion_blur, lens_blur]
    chosen_blur_effect = random.choice(blur_effects)
    degraded_img = chosen_blur_effect(degraded_img)
    return degraded_img

def apply_group_D(img):
    """Adds various types of noise (reduced)."""
    degraded_img = img.copy()
    rows, cols, ch = degraded_img.shape
    mean = 0
    std = 15
    gauss = np.random.normal(mean, std, (rows, cols, ch)).astype(np.float32)
    noisy_img = degraded_img.astype(np.float32) + gauss
    degraded_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    s_vs_p = 0.5
    amount = 0.005
    num_salt = np.ceil(amount * degraded_img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in degraded_img.shape[:2]]
    degraded_img[coords[0], coords[1]] = 255
    num_pepper = np.ceil(amount * degraded_img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in degraded_img.shape[:2]]
    degraded_img[coords[0], coords[1]] = 0
    return degraded_img

def degrade_with_group(img, group_name):
    """
    Applies degradation functions sequentially based on the selected group.

    Args:
        img (np.ndarray): Input image in BGR format.
        group_name (str): Group identifier, e.g., 'A', 'A+B', 'A+B+C', 'A+B+C+D'.

    Returns:
        np.ndarray: Degraded image in BGR format.
    """
    degraded_img = img.copy()

    if group_name == 'Acquisition & Sensor Defects':
        degraded_img = apply_group_A(degraded_img)
    elif group_name == 'Transmission & Compression artifact':
        degraded_img = apply_group_B(degraded_img)
    elif group_name == 'Blurring Effects':
        degraded_img = apply_group_C(degraded_img)
    elif group_name == 'Noise Injection':
        degraded_img = apply_group_D(degraded_img)
    elif group_name == 'Acquisition & Sensor Defects + Transmission & Compression artifact':
        degraded_img = apply_group_A(degraded_img)
        degraded_img = apply_group_B(degraded_img)
    elif group_name == 'Acquisition & Sensor Defects + Transmission & Compression artifact + Blurring Effects':
        degraded_img = apply_group_A(degraded_img)
        degraded_img = apply_group_B(degraded_img)
        degraded_img = apply_group_C(degraded_img)
    elif group_name == 'All':
        degraded_img = apply_group_A(degraded_img)
        degraded_img = apply_group_B(degraded_img)
        degraded_img = apply_group_C(degraded_img)
        degraded_img = apply_group_D(degraded_img)

    return degraded_img

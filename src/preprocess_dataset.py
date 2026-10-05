"""
Batch preprocessing script for the raw palm-image dataset.

Walks a folder of raw captures (organized as <user_folder>/<left|right>/*.jpg,
200 images per user: 100 left + 100 right) and produces the cropped,
denoised, CLAHE-equalized, skeletonized training images used by the CNN,
saved into per-user "<n>left" / "<n>right" folders.

This runs on a PC over an existing dataset, separate from the Raspberry Pi
kiosk app.
"""

import os
import glob

import cv2
import numpy as np
from matplotlib import pyplot


# --------------------------------------------------------------------------
# Configuration - EDIT THESE FOR YOUR DATASET LOCATION
# --------------------------------------------------------------------------
RAW_DATASET_DIR = "D:/PALM QUEZON/PALM QUEZON DATASET/May 5/raw/"
PROCESSED_DATASET_DIR = "D:/PALM QUEZON/PALM QUEZON DATASET/May 5/preprocess/"

IMAGES_PER_USER = 200          # 100 left + 100 right
ROI = (251, 121, 198, 185)     # x, y, w, h
CROP = (4, 4, 190, 177)        # x1, y1, w1, h1


def skeletonize(gray_image):
    img = gray_image.copy()
    skeleton = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    while cv2.countNonZero(img) > 0:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        delta = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, delta)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def extract_palm_features(bgr_image, roi=ROI, crop=CROP):
    x, y, w, h = roi
    region = bgr_image[y:y + h, x:x + w]

    denoised = cv2.fastNlMeansDenoising(region)

    kernel = np.ones((7, 7), np.uint8)
    opened = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
    yuv = cv2.cvtColor(opened, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    inverted = cv2.bitwise_not(equalized)

    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    skeleton = skeletonize(gray)
    _, binary = cv2.threshold(skeleton, 5, 255, cv2.THRESH_BINARY)

    x1, y1, w1, h1 = crop
    return binary[y1:y1 + h1, x1:x1 + w1]


class DatasetWriter:
    """Tracks which user/hand folder the next processed image belongs to.
    Mirrors the original numbering: images 1-100 -> left, 101-200 -> right,
    then it rolls over to the next user."""

    def __init__(self, output_dir, images_per_user=IMAGES_PER_USER):
        self.output_dir = output_dir
        self.images_per_user = images_per_user
        self.image_counter = 0
        self.user_counter = 0

    def _ensure_user_folders(self):
        left_dir = f"{self.output_dir}user{self.user_counter:02d}left"
        right_dir = f"{self.output_dir}user{self.user_counter:02d}right"
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)
        return left_dir, right_dir

    def save(self, image, dataset_index):
        half = self.images_per_user // 2  # 100

        if dataset_index % self.images_per_user == 0:
            self.user_counter += 1

        left_dir, right_dir = self._ensure_user_folders()

        self.image_counter += 1
        if self.image_counter > self.images_per_user:
            self.image_counter = 1

        if self.image_counter <= half:
            pyplot.imsave(f"{left_dir}/{self.image_counter}.jpg", image, cmap="gray")
        else:
            pyplot.imsave(f"{right_dir}/{self.image_counter}.jpg", image, cmap="gray")


def main():
    image_paths = glob.glob(RAW_DATASET_DIR + "*/*/*.jpg")
    writer = DatasetWriter(PROCESSED_DATASET_DIR)

    for idx, image_path in enumerate(image_paths):
        raw_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        processed = extract_palm_features(raw_image)
        writer.save(processed, idx)
        print(idx)

    print(os.listdir(RAW_DATASET_DIR))


if __name__ == "__main__":
    main()

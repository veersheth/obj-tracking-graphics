import numpy as np
import cv2 as cv
import os

input_dir = "01_images"
output_dir = "02_sifting"
os.makedirs(output_dir, exist_ok=True)

for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    img = cv.imread(input_path)

    if img is None:
        print(f"error reading: {input_path}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(gray, None)

    img_with_kp = cv.drawKeypoints(img, kp, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    output_path = os.path.join(output_dir, file_name)
    cv.imwrite(output_path, img_with_kp)
    print(f"processed: {output_path}")


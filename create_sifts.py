import cv2 as cv
import os
import numpy as np

input_dir = "01_images"
output_dir = "02_sifting"
os.makedirs(output_dir, exist_ok=True)

MAX_KEYPOINTS = 1200
THICKNESS = 2
SCALE = 0.5
MIN_RADIUS = 1
MAX_RADIUS = 20
NUM_LINES = 100
LINE_THICKNESS = 1
LINE_COLOR = (200, 200, 255)

for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    img = cv.imread(input_path)

    if img is None:
        print(f"error reading: {input_path}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp, _ = sift.detectAndCompute(gray, None)

    if not kp:
        print(f"no keypoints found: {input_path}")
        continue

    kp = sorted(kp, key=lambda x: -x.response)[:MAX_KEYPOINTS]

    responses = np.array([k.response for k in kp])
    min_r, max_r = responses.min(), responses.max()
    normalized = 255 * (responses - min_r) / (max_r - min_r + 1e-5)

    for i, point in enumerate(kp):
        x, y = int(point.pt[0]), int(point.pt[1])
        radius = int(point.size * SCALE)
        radius = max(MIN_RADIUS, min(radius, MAX_RADIUS))

        intensity = int(normalized[i])
        color = (intensity, 255 - intensity, 128)

        cv.circle(img, (x, y), radius, color, thickness=THICKNESS)

    for i in range(min(NUM_LINES, len(kp) - 1)):
        pt1 = tuple(map(int, kp[i].pt))
        pt2 = tuple(map(int, kp[i + 1].pt))
        cv.line(img, pt1, pt2, LINE_COLOR, thickness=LINE_THICKNESS)

    output_path = os.path.join(output_dir, file_name)
    cv.imwrite(output_path, img)
    print(f"processed: {output_path}")

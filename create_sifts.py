import cv2 as cv
import os

input_dir = "01_images"
output_dir = "02_sifting"
os.makedirs(output_dir, exist_ok=True)

MAX_KEYPOINTS = 400
MARKER_SIZE = 14
THICKNESS = 2

for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    img = cv.imread(input_path)

    if img is None:
        print(f"error reading: {input_path}")
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp, _ = sift.detectAndCompute(gray, None)

    kp = sorted(kp, key=lambda x: -x.response)[:MAX_KEYPOINTS]

    for point in kp:
        x, y = int(point.pt[0]), int(point.pt[1])
        top_left = (x - MARKER_SIZE // 2, y - MARKER_SIZE // 2)
        bottom_right = (x + MARKER_SIZE // 2, y + MARKER_SIZE // 2)
        cv.rectangle(img, top_left, bottom_right, (0, 255, 0), thickness=THICKNESS)

    output_path = os.path.join(output_dir, file_name)
    cv.imwrite(output_path, img)
    print(f"processed: {output_path}")

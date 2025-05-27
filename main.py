import numpy as np
import cv2 as cv
import sys

img = cv.imread(sys.argv[1])
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
 
sift = cv.SIFT_create()
kp = sift.detect(gray,None)
 
img=cv.drawKeypoints(gray,kp,img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
cv.imwrite('sifting/sift_keypoints.jpg',img)

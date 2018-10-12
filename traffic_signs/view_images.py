import cv2
import glob
import numpy as np

dirs = glob.glob("C:\\Users\\Usuario\\Desktop\\mcv-m1-project-team8\\Team8\\traffic_signs\\Dataset\\output\\maskOut\\hsv_team1_None\\*")
square_kernel = np.ones((5, 5), np.uint8)

for im_path in dirs:
    img = cv2.imread(im_path)
    # Opening
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, square_kernel)
    cv2.imshow("test", img*255)
    cv2.waitKey()
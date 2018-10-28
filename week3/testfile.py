# -*- coding: utf-8 -*-
import numpy as np
import cv2

ref = cv2.imread("./Dataset/museum_set_random/ima_000000.jpg")[:,:,0]
com = cv2.imread("./Dataset/query_devel_random/ima_000023.jpg")[:,:,0]

#reference array:
j = np.histogram(ref, bins = 80)
a = np.zeros((len(j[0]), 2))
for i, x in enumerate(j[0]):
    a[i][0] = x
    a[i][1] = j[1][0][i]
cv_array_ref = cv2.cv.fromarray(a)

a32 = cv2.cv.CreateMat(cv_array_ref.rows, cv_array_ref.cols,cv2.cv.CV_32FC1)
cv2.cv.Convert(cv_array_ref, a32)


#measured array:

jj = np.histogram(com, bins = 80)
aa = np.zeros((len(jj[0]), 2))
for ii, xx in enumerate(jj[0]):
    aa[ii][0] = xx
    aa[i][1] = jj[1][0][ii]
cv_array_meas = cv2.cv.fromarray(aa)
a322 = cv2.cv.CreateMat(cv_array_meas.rows, cv_array_meas.cols, cv2.cv.CV_32FC1)
cv2.cv.Convert(cv_array_meas, a322)


cv2.cv.CalcEMD2(a32, a322,cv2.cv.CV_DIST_L1)
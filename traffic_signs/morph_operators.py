'''This code is only a reference code from which we can select the
desired kernel structure and size from the square, rectangular, ellipsoidal
and cross options. It also has an example of erosion, dilation, opening and
closing.'''
import cv2
import numpy as np

#Defining the kernel shape and size using CV.
square_kernel = np.ones((5,5), iterations=1) #This is a 5X5 kernel.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
ellip_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE(5,5))
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

#Erosion filter with a square kernel.
erosion = cv2.erode(image,square_kernel,iterations=1)

#Dilation filter with a square kernel.
dilation = cv2.dilate(image, square_kernel, iterations=1)

#Opening filter with a square kernel.
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, square_kernel)

#Closing filter with a square kernel.
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, square_kernel)

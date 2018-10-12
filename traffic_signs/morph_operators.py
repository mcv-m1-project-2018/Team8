import cv2
import numpy as np

#Defining the kernel
square_kernel = np.ones((5,5), iterations=1) #This is a 5X5 kernel.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
ellip_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE(5,5))
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

#Erosion filter
erosion = cv2.erode(image,square_kernel,iterations=1)

#Dilation filter
dilation = cv2.dilate(image, square_kernel, iterations=1)

#Opening
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, square_kernel)

#Closing
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, square_kernel)

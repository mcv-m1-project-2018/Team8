import cv2 as cv
from tqdm import tqdm
import numpy as np


def rotate(file_names, image_path):
    n_images = len(file_names)
    
    i = 0
    while i < n_images:
        name = file_names[i]
        imageNameFile = image_path + "/" + name
        image = cv.imread(imageNameFile)

        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        k = np.ones((9,9))
        gray = cv.blur(gray,(9,9))
        edges = cv.Canny(gray,10,20)
        edges2 = cv.resize(edges,None, fx=0.3, fy=0.3)
        cv.imshow('e',edges2)


        w,_,_ = image.shape
        lines = cv.HoughLines(edges,1,np.pi/180,100)
        for l in lines:
            for rho,theta in l:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv.line(image,(x1,y1),(x2,y2),(0,255,0),2)
        
        
        image2 = cv.resize(image,None, fx=0.1, fy=0.1)
        cv.imshow('lines',image2)
        k = cv.waitKey()
        
        if k==27 or k==-1:    # Esc key or close to stop
            break
        elif k==97 and i>0:    # A to go back
            i-=1
        else:                   # Any key to go forward
            i+=1

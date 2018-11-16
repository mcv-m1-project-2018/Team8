import cv2 as cv
from tqdm import tqdm
import numpy as np
import imutils
from math import degrees, radians
from skimage import feature


def resize_keeping_ar(im, desired_width=300):
    height, width = im.shape[:2]
    factor = width/float(desired_width)
    desired_height = int(height/factor)
    imres = cv.resize(im, (desired_width, desired_height))
    return imres

def compute_angles(file_names, image_path):
    n_images = len(file_names)
    
    i = 0
    while i < n_images:
        name = file_names[i]
        imageNameFile = image_path + "/" + name
        image = cv.imread(imageNameFile)
        w,h = image.shape[:2]
        new_size = min(round(w*0.2), 300)
        image = resize_keeping_ar(image, )
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        k = np.ones((9, 9))
        print(type(gray))
        gray = cv.GaussianBlur(gray,(9,9),0)
        edges = cv.Canny(gray, 100, 130, apertureSize=3, L2gradient=True)
        edges2 = resize_keeping_ar(edges)
        cv.imshow('e', edges2)


        meanLines = []

        lines = cv.HoughLines(edges, 1, np.pi/180, 50)
        for line in lines:
            for rho, theta in line:
                if meanLines != []:
                    foundMean = False
                    for eachMean in meanLines:
                        if theta+0.1 > eachMean[2] and theta-0.1 < eachMean[2]:
                            if rho < eachMean[0] or rho > eachMean[1]:
                                eachMean[rho > eachMean[1]] = rho
                            eachMean[2] = (eachMean[2]*eachMean[3] + theta)/(eachMean[3]+1)
                            eachMean[3] += 1
                            foundMean = True
                        if foundMean:
                            break
                    if not foundMean:
                        meanLines.append([rho, rho, theta, 1])
                         
                else:
                    meanLines.append([rho, rho, theta, 1])

        rotate(meanLines, image)

        image2 = resize_keeping_ar(image)
        cv.imshow('lines', image2)
        k = cv.waitKey()

        if k == 27 or k == -1:    # Esc key or close to stop
            break
        elif k == 97 and i > 0:    # A to go back
            i -= 1
        else:                   # Any key to go forward
            i += 1


def rotate(meanLines, image, thr_angle=60, inc=5):
    # for values in meanLines:
    def thirdElement(elem):
        return elem[2]

    countLines = sum([x[3] for x in meanLines])
    rot_filter_lines = [x for x in meanLines if (degrees(x[2])<thr_angle or degrees(x[2])>(360-thr_angle))]
    while rot_filter_lines == []:
        rot_filter_lines = [x for x in meanLines if (degrees(x[2])<(thr_angle+inc) or degrees(x[2])>(360-thr_angle+inc))]
        inc += 5

    if inc > 15:
        rot_filter_lines = [(lambda x: [x[0],x[1],x[2]-radians(90),x[3]])(x) for x in rot_filter_lines]
    # thr_lines = [x for x in rot_filter_lines if x[3]>(countLines/4)]
    thr_lines = sorted(rot_filter_lines, key = lambda x: x[2], reverse=True)
    values = thr_lines[0]
    theta = values[2]

    # print(sortedLines)

    rot_theta = imutils.rotate_bound(image, 360-degrees(theta))
    res_rot_theta = resize_keeping_ar(rot_theta)
    cv.imshow('Rotated theta', res_rot_theta)
    # cv.waitKey(0)
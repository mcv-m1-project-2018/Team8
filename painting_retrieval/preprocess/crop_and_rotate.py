import cv2 as cv
from tqdm import tqdm
import numpy as np
import imutils
from math import degrees
from skimage import feature


def compute_angles(file_names, image_path):
    n_images = len(file_names)
    
    i = 0
    while i < n_images:
        name = file_names[i]
        imageNameFile = image_path + "/" + name
        image = cv.imread(imageNameFile)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        k = np.ones((9, 9))
        print(type(gray))
        gray = cv.blur(gray, (9, 9))
        edges = cv.Canny(gray, 90, 130, apertureSize=3)
        # edges = feature.canny(gray)
        edges2 = cv.resize(edges, None, fx=0.3, fy=0.3)
        cv.imshow('e', edges2)


        meanLines = []

        lines = cv.HoughLines(edges, 1, np.pi/180, 100)
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
        # print(meanLines)

        rotate(meanLines, image)

        image2 = cv.resize(image, None, fx=0.1, fy=0.1)
        cv.imshow('lines', image2)
        k = cv.waitKey()

        if k == 27 or k == -1:    # Esc key or close to stop
            break
        elif k == 97 and i > 0:    # A to go back
            i -= 1
        else:                   # Any key to go forward
            i += 1


def rotate(meanLines, image):
    # for values in meanLines:
    def thirdElement(elem):
        return elem[2]

    # sortedLines = sorted(meanLines, key=thirdElement, reverse=True)
    sortedLines = sorted(meanLines, key = lambda x: x[3], reverse=True)
    values = sortedLines[0]
    theta = values[2]

    print(sortedLines)

    rot_theta = imutils.rotate_bound(image, 360-degrees(theta))
    res_rot_theta = cv.resize(rot_theta, None, fx=0.1, fy=0.1)
    cv.imshow('Rotated theta', res_rot_theta)
    cv.waitKey(0)

    # rot_rho1 = imutils.rotate_bound(image, 360-degrees(rho1))
    # res_rot_rho1 = cv.resize(rot_rho1, None, fx=0.1, fy=0.1)
    # rot_rho2 = imutils.rotate_bound(image, 360-degrees(rho2))
    # res_rot_rho2 = cv.resize(rot_rho2, None, fx=0.1, fy=0.1)
    # cv.imshow('Rotated without losing rho1', res_rot_rho1)
    # cv.imshow('Rotated without losing rho2', res_rot_rho2)
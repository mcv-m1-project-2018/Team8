import cv2 as cv
from tqdm import tqdm
import numpy as np
import imutils
from math import degrees, radians, sin, cos
from skimage import feature


def resize_keeping_ar(im, desired_width=300):
    height, width = im.shape[:2]
    factor = width/float(desired_width)
    desired_height = int(height/factor)
    imres = cv.resize(im, (desired_width, desired_height))
    return imres

def intersection(line1, line2, intersectThrUp = 95, intersectThrBottom = 85):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    diff = abs(degrees(theta1)-degrees(theta2))
    if(diff >= intersectThrBottom and diff <= intersectThrUp):
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
    else:
        x0, y0 = -1, -1
    return [x0, y0, rho1, theta1, rho2, theta2]

def filterPoints(point_list, imgWidth, imgHeight):
    brPoint = [0,0, 0, 0, 0, 0]
    tlPoint = [imgWidth,imgHeight, 0, 0, 0, 0]

    trPoint = [0,0, 0, 0, 0, 0]
    blPoint = [0,0, 0, 0, 0, 0]

        #getting the tl and br points
    for x, y, rho1, theta1, rho2, theta2  in point_list:

        orgValue = brPoint[0] + brPoint[1]
        currentValue = x + y
        if(currentValue > orgValue):
            brPoint = [x,y, rho1, theta1, rho2, theta2]

        orgValue = tlPoint[0] + tlPoint[1]
        currentValue = x + y
        if(currentValue < orgValue):
            tlPoint = [x,y, rho1, theta1, rho2, theta2]

    #once we have the tl and br points we can get the other points with the line intersection
    trPoint = intersection([brPoint[2],brPoint[3]],[tlPoint[4],tlPoint[5]])
    blPoint = intersection([brPoint[4],brPoint[5]],[tlPoint[2],tlPoint[3]])

    return [tlPoint, trPoint, blPoint, brPoint]



def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, (rho1, _, theta1, _) in enumerate(lines[:-1]):
        for (rho2, _, theta2, _) in lines[i+1:]:
            point = intersection([rho1, theta1], [rho2, theta2])
            if(point[0] != -1 and point[1] != -1):
                intersections.append(point) 
        for (_ , rho2, theta2, _) in lines:
            point = intersection([rho1,theta1], [rho2,theta2])
            if(point[0] != -1 and point[1] != -1):
                intersections.append(point) 
            
    for i, (_ , rho1, theta1, _) in enumerate(lines[:-1]):
        for (_ , rho2, theta2, _) in lines[i+1:]:
            point = intersection([rho1, theta1], [rho2, theta2])
            if(point[0] != -1 and point[1] != -1):
                intersections.append(point) 
        for (rho2 , _, theta2, _) in lines:
            point = intersection([rho1,theta1], [rho2,theta2])
            if(point[0] != -1 and point[1] != -1):
                intersections.append(point) 

    return intersections


def fill_holes(im):
    _, contours, _ = cv.findContours(im, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv.drawContours(im, [cnt], 0, 255, -1)
    return im


def morph_method1(im):
    imggray = im.copy()

    morph = imggray.copy()


    vk = lambda x : np.ones((x, 1), np.uint8)
    hk = lambda x : np.ones((1, x), np.uint8)
    kk = lambda x : np.ones((x, x), np.uint8)

    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kk(15))
    morph = fill_holes(morph)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, kk(5))
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, kk(10))
    
    imagen = morph
    #imagen = fill_holes(morph)
    
    #rows, cols = imagen.shape
    #M = np.float32([[1, 0, -5], [0, 1, -5]])
    #imagen = cv.warpAffine(imagen, M, (cols, rows))
    return imagen


def compute_angles(file_names, image_path):
    n_images = len(file_names)

    i = 0
    while i < n_images:
        name = file_names[i]
        imageNameFile = image_path + "/" + name
        image = cv.imread(imageNameFile)
        w,h = image.shape[:2]
        new_size = min(round(w*0.2), 300)
        image = resize_keeping_ar(image)
        # hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        # up_bound = np.array([255,255,70])
        # low_bound = np.array([0,0,0])
        # gray = cv.inRange(hsv_image, low_bound, up_bound)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #k = np.ones((9, 9))
        gray = cv.GaussianBlur(gray, (11, 11), 0)
        resized = resize_keeping_ar(gray)
        edges = cv.Canny(resized, 0, 40, apertureSize=3, L2gradient=True)
        cv.imshow('Canny', edges)
        morph_img = morph_method1(edges)
        fh_img = fill_holes(morph_img)
        edges2 = cv.Canny(fh_img, 60, 80, apertureSize=3, L2gradient=True)
        cv.imshow('Canny_2', edges2)

        meanLines = []

        lines = cv.HoughLines(edges2, 1, np.pi/180, 50)
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

        rotate(meanLines, image.copy())

        points = segmented_intersections(meanLines)

        points = filterPoints(points, image.shape[1], image.shape[0])

        showMeanLinesAndIntersections(meanLines, points, image)

        image2 = resize_keeping_ar(image.copy())
        # cv.imshow('lines', image2)
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
    rot_filter_lines = [x for x in meanLines if (degrees(x[2])<thr_angle or degrees(x[2]) > (360-thr_angle))]
    while rot_filter_lines == []:
        rot_filter_lines = [x for x in meanLines if (degrees(x[2])<(thr_angle+inc) or degrees(x[2]) > (360-thr_angle+inc))]
        inc += 5

    if inc > 15:
        rot_filter_lines = [(lambda x: [x[0], x[1], x[2]-radians(90), x[3]])(x) for x in rot_filter_lines]
    # thr_lines = [x for x in rot_filter_lines if x[3]>(countLines/4)]
    thr_lines = sorted(rot_filter_lines, key=lambda x: x[2], reverse=True)
    values = thr_lines[0]
    theta = values[2]

    # print(sortedLines)

    # showMeanLinesAndIntersections(thr_lines,[],image)

    rot_theta = imutils.rotate_bound(image, 360-degrees(theta))
    res_rot_theta = resize_keeping_ar(rot_theta)
    cv.imshow('Rotated theta', res_rot_theta)
    # cv.waitKey(0)

def showMeanLinesAndIntersections(meanLines, points, image):
    if meanLines is not None:
        for (rho1, rho2, theta, _) in meanLines:
            rho = rho1
            a = cos(theta)
            b = sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*a))
            pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*a))
            cv.line(image, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
            rho = rho2
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 10000*(-b)), int(y0 + 10000*(a)))
            pt2 = (int(x0 - 10000*(-b)), int(y0 - 10000*(a)))
            cv.line(image, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
    for x,y, _, _, _, _ in points:
        cv.circle(image, (x, y), 5, (255, 0, 0), thickness=-1)
    image = resize_keeping_ar(image, 300)
    cv.imshow('lines and points', image)
    k = cv.waitKey()

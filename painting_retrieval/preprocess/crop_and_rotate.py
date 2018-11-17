import cv2 as cv
from tqdm import tqdm
import numpy as np
import imutils
from math import degrees, radians, sin, cos
from skimage import feature
#import bounding_box_utils
import preprocess.find_largest_rectangle as find_largest_rectangle
from preprocess.utils import add_margin, get_center_diff

def resize_keeping_ar(im, desired_width=300):
    height, width = im.shape[:2]
    factor = width/float(desired_width)
    desired_height = int(height/factor)
    imres = cv.resize(im, (desired_width, desired_height))
    return imres, factor

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
    if(brPoint[3] != tlPoint[5]):
        trPoint = intersection([brPoint[2],brPoint[3]],[tlPoint[4],tlPoint[5]])
        blPoint = intersection([brPoint[4],brPoint[5]],[tlPoint[2],tlPoint[3]])
    else:
        trPoint = intersection([brPoint[2],brPoint[3]],[tlPoint[2],tlPoint[3]])
        blPoint = intersection([brPoint[4],brPoint[5]],[tlPoint[4],tlPoint[5]])
        if(abs(trPoint[2]) < abs(blPoint[2])):
            trPoint, blPoint = blPoint, trPoint

    return tlPoint, trPoint, brPoint, blPoint



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


#    vk = lambda x : np.ones((x, 1), np.uint8)
#    hk = lambda x : np.ones((1, x), np.uint8)
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

def morph_method2(im):
    imggray = im.copy()

    morph = imggray.copy()


    vk = lambda x : np.ones((x, 1), np.uint8)
    hk = lambda x : np.ones((1, x), np.uint8)
#    kk = lambda x : np.ones((x, x), np.uint8)
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, vk(23))
    morph = cv.morphologyEx(morph, cv.MORPH_CLOSE, hk(23))
    morph = fill_holes(morph)
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, hk(40))
    morph = cv.morphologyEx(morph, cv.MORPH_OPEN, vk(40))
    
    #### DO CCL ###
#    bb_list = bounding_box_utils(morph)
#    for bb in bb_list:
        
    imagen = morph
    #imagen = fill_holes(morph)
    
    #rows, cols = imagen.shape
    #M = np.float32([[1, 0, -5], [0, 1, -5]])
    #imagen = cv.warpAffine(imagen, M, (cols, rows))
    return imagen

def get_contours1(im, debug=False, add_str_debug=""):
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (11, 11), 0)
    edges = cv.Canny(gray, 0, 40, apertureSize=3, L2gradient=True)
    morph_img = morph_method1(edges)
    fh_img = fill_holes(morph_img)
    edges2 = cv.Canny(fh_img, 60, 80, apertureSize=3, L2gradient=True)
    edges2 = cv.dilate(edges2,(3,3),iterations = 1)
    if(debug): 
        cv.imshow('Canny'+add_str_debug, edges)
        cv.imshow('fill_holes'+add_str_debug, fh_img)
        cv.imshow('Canny_2'+add_str_debug, edges2)
    
    return edges2, fh_img

def get_contours2(filled, debug=False, add_str_debug=""):
#    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#    gray = cv.GaussianBlur(edges, (11, 11), 0)
#    edges = cv.Canny(gray, 0, 40, apertureSize=3, L2gradient=True)
#    morph_img = morph_method2(edges)
#    fh_img = fill_holes(morph_img)
    madd = 1
    edges = add_margin(filled, madd).astype(np.uint8)
    edges2 = cv.Canny(edges, 60, 80, apertureSize=3, L2gradient=True)
    edges2 = cv.dilate(edges2,(3,3),iterations = 1)
    edges2 = edges2[madd:-(madd+1), madd:-(madd+1)]
    if(debug): 
#        cv.imshow('Canny'+add_str_debug, edges)
#        cv.imshow('fill_holes'+add_str_debug, fh_img)
        cv.imshow('Canny_2'+add_str_debug, edges2)
    
    return edges2

def get_hough_lines(edges):
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
    return meanLines

#from math import sin, cos, radians

def rotate_point(point, angle, center_point=(0, 0), convert_ints = True):
    """Rotates a point around center_point(origin by default)
    Angle is in degrees.
    Rotation is counter-clockwise
    """
    angle_rad = radians(angle % 360)
    # Shift the point so that center_point becomes the origin
    new_point = (point[0] - center_point[0], point[1] - center_point[1])
    new_point = (new_point[0] * cos(angle_rad) - new_point[1] * sin(angle_rad),
                 new_point[0] * sin(angle_rad) + new_point[1] * cos(angle_rad))
    # Reverse the shifting we have done
    x = new_point[0] + center_point[0]
    y = new_point[1] + center_point[1]
    if(convert_ints):
        x = int(x)
        y = int(y)
    new_point = (x, y)
    return new_point

def calc_points_morphologically(image, edges, meanLines, debug = False):
    edges_rot, angle = rotate(meanLines, edges)
    edges_rot_fill = morph_method2(edges_rot)
    max_size, pos = find_largest_rectangle.max_size(edges_rot_fill, \
                                                    np.max(edges_rot_fill))
    
    
#    img, angle = rotate(meanLines, image.copy())
    
    
    
#            img = np.dstack([edges_rot_fill]*3)
    minY, maxY = pos[0]-max_size[0], pos[0]
    minX, maxX = pos[1]-max_size[1], pos[1]
    c1 = (minX, minY)
    c2 = (minX, maxY)
    c3 = (maxX, minY)
    c4 = (maxX, maxY)

#            center_point = ( w/2-(w-nw), h/2-(h-nh))
    c_diff = get_center_diff(edges_rot_fill, image)
    rot_h, rot_w = edges_rot_fill.shape
    rot_cp = (rot_w/2, rot_h/2)
    nc1 = rotate_point(c1, -angle, rot_cp)
    nc2 = rotate_point(c2, -angle, rot_cp)
    nc3 = rotate_point(c3, -angle, rot_cp)
    nc4 = rotate_point(c4, -angle, rot_cp)
    rnc1 = (nc1[0]+c_diff[0], nc1[1]+c_diff[1])
    rnc2 = (nc2[0]+c_diff[0], nc2[1]+c_diff[1])
    rnc3 = (nc3[0]+c_diff[0], nc3[1]+c_diff[1])
    rnc4 = (nc4[0]+c_diff[0], nc4[1]+c_diff[1])
    TL, BL, TR, BR = rnc1, rnc2, rnc3, rnc4
    
    if(debug):
        img = imutils.rotate_bound(image.copy(), angle)
        
        cv.imshow('Painting', edges_rot_fill)
        
        color = ( 0,0,255)
        cv.line(img, (0, minY), (rot_w, minY), color)
        cv.line(img, (minX, 0), (minX, rot_h), color)
        cv.line(img, (0, maxY), (rot_w, maxY), color)
        cv.line(img, (maxX, 0), (maxX, rot_h), color)
        color = (0, 255, 0)
        radius = 2
        thick = 3
        cv.circle(img, c1, radius, color, thickness=thick)
        cv.circle(img, c2, radius, color, thickness=thick)
        cv.circle(img, c3, radius, color, thickness=thick)
        cv.circle(img, c4, radius, color, thickness=thick)
        cv.imshow('Rectangle', img)
        
        img2 = image.copy()
        cv.circle(img2, rnc1, radius, color, thickness=thick)
        cv.circle(img2, rnc2, radius, color, thickness=thick)
        cv.circle(img2, rnc3, radius, color, thickness=thick)
        cv.circle(img2, rnc4, radius, color, thickness=thick)
        cv.imshow('Unrotated rectangle', img2)
                
    return TL, TR, BR, BL
            
def compute_angles(file_names, image_path, debug = True):
    method_extract_bb = "*"
    n_images = len(file_names)

    i = 0
    while i < n_images:
        name = file_names[i]
        imageNameFile = image_path + "/" + name
        image_original = cv.imread(imageNameFile)
        w,h = image_original.shape[:2]
#        new_size = min(round(w*0.2), 300)
        image, factor = resize_keeping_ar(image_original.copy())
        increase_point = lambda p: (int(p[0]*factor), int(p[1]*factor))
        edges, fh_image = get_contours1(image, debug)
#        edges, morph_img = get_contours3(image, True)
        meanLines = get_hough_lines(edges)
        
        
        if(method_extract_bb in ["morphologically","*"]):
            tlp, trp, brp, blp = calc_points_morphologically(image, edges, \
                                                             meanLines, debug)
        if(method_extract_bb in ["hough","*"]):
            points = segmented_intersections(meanLines)
            tlp, trp, brp, blp = filterPoints(points, image.shape[1], image.shape[0])
            points = [tlp, trp, brp, blp]
            tlp, trp, brp, blp = tlp[:2], trp[:2], brp[:2], blp[:2]
            if(debug): showMeanLinesAndIntersections(meanLines, points, image.copy(), " H")
        if(method_extract_bb in ["hough_rotated","*"]):
            __, angle = rotate(meanLines, edges)
            fh_image_rot = imutils.rotate_bound(fh_image.copy(), angle)
            image_rot = imutils.rotate_bound(image.copy(), angle)
            rotated_filled = morph_method2(fh_image_rot)
            edges_rotated_filled = get_contours2(rotated_filled, True, " HR")
            meanLines = get_hough_lines(edges_rotated_filled)
            
            points = segmented_intersections(meanLines)
            tlp, trp, brp, blp = filterPoints(points, image.shape[1], image.shape[0])
            points = [tlp, trp, brp, blp]
            
            c_diff = get_center_diff(edges_rotated_filled, image)
#            c1, c2, c3, c4 = tlp[:2], trp[:2], brp[:2], blp[:2]
            
            # ORDER POINTS
#            points_list = [c1,c2,c3,c4]
            sortX = lambda x: x[0]
            sortY = lambda x: x[1]
            orderedX = sorted(points, key=sortX)
#            orderedY = sorted(points_list, key=sortY)
            TL = sorted(orderedX[:2], key=sortY)[0]
            DL = sorted(orderedX[:2], key=sortY)[1]
            TR = sorted(orderedX[2:], key=sortY)[0]
            DR = sorted(orderedX[2:], key=sortY)[1]
            tlp, trp, brp, blp = TL, TR, DR, DL
            points = [tlp, trp, brp, blp]
            ##############
            rot_h, rot_w = edges_rotated_filled.shape
            rot_cp = (rot_w/2, rot_h/2)
    
            nc1 = rotate_point(tlp, -angle, rot_cp)
            nc2 = rotate_point(trp, -angle, rot_cp)
            nc3 = rotate_point(brp, -angle, rot_cp)
            nc4 = rotate_point(blp, -angle, rot_cp)
            rnc1 = (nc1[0]+c_diff[0], nc1[1]+c_diff[1])
            rnc2 = (nc2[0]+c_diff[0], nc2[1]+c_diff[1])
            rnc3 = (nc3[0]+c_diff[0], nc3[1]+c_diff[1])
            rnc4 = (nc4[0]+c_diff[0], nc4[1]+c_diff[1])
            
#            points [[x[0],x[1] + y[2:]] for x,y in zip([rnc1, rnc2, rnc3, rnc4])]
#            points = [rnc1, rnc2, rnc3, rnc4]
#            tlp, trp, brp, blp = tlp[:2], trp[:2], brp[:2], blp[:2]
            
            
            if(debug): showMeanLinesAndIntersections(meanLines, points, image_rot, " HR")

        
        tlp = increase_point(tlp)
        trp = increase_point(trp)
        brp = increase_point(brp)
        blp = increase_point(blp)
 
        # cv.imshow('lines', image2)
        if(debug):
            k = cv.waitKey()
    
            if k == 27 or k == -1:    # Esc key or close to stop
                break
            elif k == 97 and i > 0:    # A to go back
                i -= 1
            else:                   # Any key to go forward
                i += 1
        else:
            i+=1

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
    angle = 360-degrees(theta)
    rot_theta = imutils.rotate_bound(image, angle )
#    res_rot_theta = resize_keeping_ar(rot_theta)
    return rot_theta, angle
    # cv.waitKey(0)

def showMeanLinesAndIntersections(meanLines, points, image,added_title=""):
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
    colours = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
    i = 0
    for x,y, _, _, _, _ in points:
        cv.circle(image, (x, y), 5, colours[i], thickness=-1)
        i-=1
    cv.imshow('lines and points'+added_title, image)


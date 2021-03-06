import cv2 as cv
from tqdm import tqdm
import numpy as np
import imutils
from math import degrees, radians, sin, cos
from skimage import feature
import pickle as pckl
import preprocess.find_largest_rectangle as find_largest_rectangle
from preprocess.utils import get_center_diff, rotate_point , resize_keeping_ar, \
                            correct_point
from preprocess.morphology import morph_method2, get_contours1, get_contours2

def saveCroppingArray(save_path, croppingArray):
    pckl_file = open(save_path,"wb")
    pckl.dump(croppingArray,pckl_file)
    pckl_file.close()

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

def calc_points_morphologically(image, fh_img, meanLines, debug = False):
    edges_rot, angle = rotate(meanLines, fh_img)
    edges_rot_fill = morph_method2(edges_rot)
    max_size, pos = find_largest_rectangle.max_size(edges_rot_fill, \
                                                    np.max(edges_rot_fill))
    
    minY, maxY = pos[0]-max_size[0], pos[0]
    minX, maxX = pos[1]-max_size[1], pos[1]
    c1 = (minX, minY)
    c2 = (minX, maxY)
    c3 = (maxX, minY)
    c4 = (maxX, maxY)

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
        
        color = (0,0,255)
        cv.line(img, (0, minY), (rot_w, minY), color)
        cv.line(img, (minX, 0), (minX, rot_h), color)
        cv.line(img, (0, maxY), (rot_w, maxY), color)
        cv.line(img, (maxX, 0), (maxX, rot_h), color)
#        color = (0, 255, 0)
#        radius = 2
#        thick = 3
#        cv.circle(img, c1, radius, color, thickness=thick)
#        cv.circle(img, c2, radius, color, thickness=thick)
#        cv.circle(img, c3, radius, color, thickness=thick)
#        cv.circle(img, c4, radius, color, thickness=thick)
#        cv.imshow('Rectangle', img)
        
#        img2 = image.copy()
#        cv.circle(img2, rnc1, radius, color, thickness=thick)
#        cv.circle(img2, rnc2, radius, color, thickness=thick)
#        cv.circle(img2, rnc3, radius, color, thickness=thick)
#        cv.circle(img2, rnc4, radius, color, thickness=thick)
#        cv.imshow('Unrotated rectangle', img2)
                
    return TL, TR, BR, BL

def load_angle_points(filename):
    with open(filename, 'rb') as pickle_file:
        index = pckl.load(pickle_file)
    angle, tlp, trp, brp, blp = index
    return angle, tlp, trp, brp, blp

def save_angle_points(filename, angle, tlp, trp, brp, blp):
    index = [angle, tlp, trp, brp, blp]
    # Dump the keypoints
    f = open(filename, "wb")
    pckl.dump(index,f)
    f.close()
    
import os
angles_folder_name = "/angles/"
def compute_angles(file_names, image_path, cropping_method = "morphologically", debug = False):
    angles_folder = image_path+angles_folder_name
    if not os.path.exists(angles_folder):
        os.makedirs(angles_folder)
        
    n_images = len(file_names)
    cropping_list = []
    i = 0
    while i < n_images:
        name = file_names[i]
        print(i, name)
        filename = angles_folder+cropping_method+"_"+name+".txt"
        loaded = False
        if(os.path.isfile(filename) and not debug):
            angle, tlp, trp, brp, blp = load_angle_points(filename)
            loaded = True
        else:
            
            imageNameFile = image_path + "/" + name
            image_original = cv.imread(imageNameFile)
            orig_h,orig_w = image_original.shape[:2]
    #        new_size = min(round(w*0.2), 300)
            image, factor = resize_keeping_ar(image_original.copy())
            increase_point = lambda p: (int(p[0]*factor), int(p[1]*factor))
            edges, fh_image = get_contours1(image, debug)
    #        edges, morph_img = get_contours3(image, True)
            meanLines = get_hough_lines(edges)
            _, angle = rotate(meanLines, edges)
            
            
            if(cropping_method in ["morphologically","*"]):
                tlp, trp, brp, blp = calc_points_morphologically(image, fh_image, \
                                                                 meanLines, debug)
                # print(tlp, trp, brp, blp)
            if(cropping_method in ["hough","*"]):
                points = segmented_intersections(meanLines)
                tlp, trp, brp, blp = filterPoints(points, image.shape[1], image.shape[0])
                points = [tlp, trp, brp, blp]
                tlp, trp, brp, blp = tlp[:2], trp[:2], brp[:2], blp[:2]
    
                if(debug): showMeanLinesAndIntersections(meanLines, points, image.copy(), " H")
            if(cropping_method in ["hough_rotated","*"]):
                __, angle = rotate(meanLines, edges)
                fh_image_rot = imutils.rotate_bound(fh_image.copy(), angle)
                image_rot = imutils.rotate_bound(image.copy(), angle)
                rotated_filled = morph_method2(fh_image_rot)
                edges_rotated_filled = get_contours2(rotated_filled, debug, " HR")
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
    
                # print(rnc1, rnc2, rnc3, rnc4)
    
                tlp, trp, brp, blp = rnc1, rnc2, rnc3, rnc4
                
    #            points [[x[0],x[1] + y[2:]] for x,y in zip([rnc1, rnc2, rnc3, rnc4])]
    #            points = [rnc1, rnc2, rnc3, rnc4]
    #            tlp, trp, brp, blp = tlp[:2], trp[:2], brp[:2], blp[:2]
                
                
                if(debug): showMeanLinesAndIntersections(meanLines, points, image_rot, " HR")
            
            if(debug): show_points(image, [tlp, trp, brp, blp])
            fix_point = lambda x: correct_point(increase_point(x), orig_w, orig_h )
            tlp = fix_point(tlp)
            trp = fix_point(trp)
            brp = fix_point(brp)
            blp = fix_point(blp)
            save_angle_points(filename, angle, tlp, trp, brp, blp)
            if(debug): show_points(image_original, [tlp, trp, brp, blp], resize=True)
            
        stored_angle = angle-180
        cropping_list.append([stored_angle,[tlp,trp,brp,blp]])

        if(debug and not loaded):
            k = cv.waitKey()
    
            if k == 27 or k == -1:    # Esc key or close to stop
                break
            elif k == 97 and i > 0:    # A to go back
                i -= 1
            else:                   # Any key to go forward
                i += 1
        else:
            i+=1
    return cropping_list

def rotate(meanLines, image, thr_angle=60, inc=5):
    def thirdElement(elem):
        return elem[2]
    
    rot_filter_lines = [x for x in meanLines if (degrees(x[2])<thr_angle or degrees(x[2]) > (360-thr_angle))]
    while rot_filter_lines == []:
        rot_filter_lines = [x for x in meanLines if (degrees(x[2])<(thr_angle+inc) or degrees(x[2]) > (360-thr_angle+inc))]
        inc += 5

    if inc > 15:
        rot_filter_lines = [(lambda x: [x[0], x[1], x[2]-radians(90), x[3]])(x) for x in rot_filter_lines]
    thr_lines = sorted(rot_filter_lines, key=lambda x: x[2], reverse=True)
    values = thr_lines[0]
    theta = values[2]

    angle = 360-degrees(theta)
    rot_theta = imutils.rotate_bound(image, angle)
    return rot_theta, angle

def show_points(img, points, added_title="", resize=False):
    colours = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
    i = 0
    h, w = img.shape[:2]
    radius=int(max(w/100,5))
    thick=int(max(w/100-1,5-1))
    for x,y in points:
        cv.circle(img, (x, y), radius, colours[i], thickness=thick)
        i-=1
    if(resize): img = resize_keeping_ar(img)[0]
    cv.imshow('points'+added_title, img)   
    
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
    pts = [(p[0], p[1]) for p in points]
    show_points(image, pts,added_title)

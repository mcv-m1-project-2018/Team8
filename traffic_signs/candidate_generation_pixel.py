#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from skimage import color
import cv2 as cv

from candidate_generation_window import reduce_winds_sizes

def masks_rgb(im):
    """
    Performs RGB pixel candidate selection
    * Inputs:
    - im = RGB image
    *Outputs:
    - mskr, mskb: mask for Red pixels, mask for Blue pixels
    """
    image = im[:,:,:]

    # filter for red signals:
    mskr = image[:,:,0] > 70
    mskr = mskr*(image[:,:,1] < 50)
    mskr = mskr*(image[:,:,2] < 50)

    #blue colored signals
    mskb = image[:,:,0] < 50
    mskb = mskb*(image[:,:,1] < 100)
    mskb = mskb*(image[:,:,2] > 60)

    return mskr, mskb


def mask_luv(im):
    """
    Performs Luv pixel candidate selection
    * Inputs:
    - im = Luv image
    *Outputs:
    - mskr, mskb: mask for Red pixels, mask for Blue pixels
    """
    image = im[:,:,:]
    image = cv.cvtColor(image,cv.COLOR_RGB2Luv)

    mskb = image[:,:,2] > 68
    mskb = mskb*(image[:,:,2] < 114)

    mskr = image[:,:,2] > 127
    mskr = mskr*(image[:,:,2] < 157)
    return mskr, mskb

def mask_luv_grayWorld(im):
  image = im[:,:,:]
  image = cv.cvtColor(image,cv.COLOR_RGB2Luv)

  mskb = image[:,:,2] > 20
  mskb = mskb*(image[:,:,2] < 90)
  mskb = mskb*image[:,:,1] > 20
  mskb = mskb*(image[:,:,1] < 100)

  mskr = image[:,:,2] > 127
  mskr = mskr*(image[:,:,2] < 157)
  return mskr, mskb

def mask_lab(im):
    """
    Performs Lab pixel candidate selection
    * Inputs:
    - im = Lab image
    *Outputs:
    - mskr, mskb: mask for Red pixels, mask for Blue pixels
    """
    image = im[:,:,:]

    image = cv.cvtColor(image,cv.COLOR_RGB2Lab)

    mskb = image[:,:,2] < 115
    mskb = mskb*(image[:,:,0] > 40)
    mskb = mskb*(image[:,:,1] < 200)
    mskb = mskb*(image[:,:,2] > 35)

    mskr = image[:,:,1] > 140
    mskr = mskr*(image[:,:,0] > 20)
    mskr = mskr*(image[:,:,0] < 220)
    mskr = mskr*(image[:,:,2] < 150)
    mskr = mskr*(image[:,:,2] > 125)

    return mskr, mskb

def mask_hsv(im):
    """
    Performs HSV pixel candidate selection
    * Inputs:
    - im = RGB image
    *Outputs:
    - mskr, mskb: mask for Red pixels, mask for Blue pixels
    """
    # image = im[:,:,:]

    # image = cv.cvtColor(image,cv.COLOR_RGB2HSV)

    # mskr = (((image[:,:,0] < 15) | (image[:,:,0] > 350)))
    # mskr = mskr*(image[:,:,1] > 70)
    # mskr = mskr*(image[:,:,2] > 30)


    # mskb = ((image[:,:,0] > 200) & (image[:,:,0] < 255))
    # mskb = mskb*(image[:,:,1] > 70)
    # mskb = mskb*(image[:,:,2] > 30)

    hsv_im = color.rgb2hsv(im)

    mask_red = (((hsv_im[:,:,0] < 0.027) | (hsv_im[:,:,0] > 0.93)))
    mask_blue = ((hsv_im[:,:,0] > 0.55) & (hsv_im[:,:,0] < 0.75))


    return mask_red, mask_blue


#############################################
def candidate_generation_pixel_rgb(im):
    """
    Performs WHOLE RGB pixel candidate selection
    * Inputs:
    - im = RGB image
    *Outputs:
    - pixel_candidates: pixels that are possible part of a signal
    """
    mskr, mskb = masks_rgb(im)
    return mskr+mskb

def candidate_generation_pixel_hsv_team1(im):
    """
    Performs WHOLE HSV pixel candidate selection
    * Inputs:
    - im = HSV image
    *Outputs:
    - pixel_candidates: pixels that are possible part of a signal
    """
    mskr,mskb = mask_hsv(im)
    return mskr+mskb

def candidate_generation_pixel_lab(im):
    """
    Performs WHOLE Lab pixel candidate selection
    * Inputs:
    - im = Lab image
    *Outputs:
    - pixel_candidates: pixels that are possible part of a signal
    """
    mskr, mskb = mask_lab(im)
    return mskr+mskb

def candidate_generation_pixel_luv(im):
    """
    Performs Luv pixel candidate selection
    * Inputs:
    - im = Luv image
    *Outputs:
    - pixel_candidates: pixels that are possible part of a signal
    """

    mskr, mskb = mask_luv(im)
    return mskr, mskb

def candidate_generation_GW_pixel_luv(im):
    """
    Performs GrayWorld Luv pixel candidate selection
    * Inputs:
    - im = Luv image
    *Outputs:
    - pixel_candidates: pixels that are possible part of a signal
    """

    mskr, mskb = mask_luv_grayWorld(im)

    return mskr, mskb


def candidate_generation_pixel_normrgb(im): 
    """
    Performs WHOLE normrgb pixel candidate selection
    * Inputs:
    - im = normrgbs image
    *Outputs:
    - mskr:  mask for Red pixels
    """
    im = preprocess_normrgb(im)

    # filter to get noise:
    mskr = im[:,:,0] > 20
    mskr = mskr*(im[:,:,1] > 20)
    mskr = mskr*(im[:,:,2] > 20)

    return mskr

def candidate_generation_pixel_hsvb_rgbr(im):
    mskr, _ = masks_rgb(im)
    _ , mskb = mask_hsv(im)
    return mskb+mskr


def candidate_generation_pixel_luvb_rgbr(im): 
    mskr, _ = masks_rgb(im)
    _ , mskb = candidate_generation_pixel_luv(im)
    return mskb+mskr

def candidate_generation_pixel_luvb_hsvr(im):
    mskr, _ = mask_hsv(im)
    _ , mskb = candidate_generation_pixel_luv(im)

    return mskb+mskr
def candidate_generation_pixel_normrgb_luvb_rgbr(im):
    noiseMask = candidate_generation_pixel_normrgb(im)
    msk = candidate_generation_pixel_luvb_rgbr(im)
    msk = msk*(np.logical_not(noiseMask))
    return msk

###############################
def preprocess_blur(im):
    """
    Performs Blur to the image
    * Inputs:
    - im = skimage.io image
    *Outputs:
    - im = image blurred
    """
    window_mean = 5
    blurred_img = cv.blur(im,(window_mean, window_mean))
    return blurred_img

def preprocess_normrgb(im):
    """
    Performs Normalizes RGB color of the image
    * Inputs:
    - im = skimage.io image
    *Outputs:
    - im = image with rgb color normalized
    """
    # convert input image to the normRGB color space

    normrgb_im = np.zeros(im.shape)
    eps_val = 0.00001
    norm_factor_matrix = im[:,:,0] + im[:,:,1] + im[:,:,2] + eps_val

    normrgb_im[:,:,0] = im[:,:,0] / norm_factor_matrix
    normrgb_im[:,:,1] = im[:,:,1] / norm_factor_matrix
    normrgb_im[:,:,2] = im[:,:,2] / norm_factor_matrix

    normrgb_im = normrgb_im.astype(np.uint8)

    return normrgb_im

def preprocess_whitePatch(im):
    """
    Performs WhitePatch to the image
    * Inputs:
    - im = skimage.io image
    *Outputs:
    - im = image with WhitePatch filter applied
    """
    bmax, gmax, rmax = np.amax(np.amax(im,axis=0),axis=0)

    alpha = gmax/rmax
    beta = gmax/bmax

    red = im[:,:,2]
    red = alpha * red
    red[red > 255] = 255
    im[:,:,2] = red

    blue = im[:,:,0]
    blue = beta * blue
    blue[blue > 255] = 255
    im[:,:,0] = blue

    return im

def preprocess_grayWorld(im):
    """
    Performs GrayWorld to the image
    * Inputs:
    - im = skimage.io image
    *Outputs:
    - im = image with GrayWorld filter applied
    """
    bmean, gmean, rmean = np.mean(np.mean(im,axis=0),axis=0)

    alpha = gmean/rmean
    beta = gmean/bmean

    red = im[:,:,2]
    red = alpha * red
    red[red > 255] = 255
    im[:,:,2] = red

    blue = im[:,:,0]
    blue = beta * blue
    blue[blue > 255] = 255
    im[:,:,0] = blue

    return im

def preprocess_neutre(im):  
    """
    Local color neutralizator
    * Inputs:
    - im = skimage.io image
    *Outputs:
    - im = imatge amb color neutralitzat
    """
    r, g, b = im[:,:,0], im[:,:,1], im[:,:,2]

    [x, y, z] = im.shape
    sz = int(x/10)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (sz, sz))
    capes = [r,g,b]
    noms = ["r","g","b"]
    cv.imshow('original',im)
    res = ()
    for i, cn in enumerate(zip(capes, noms)):
        capa = cn[0]
        nom = cn[1]
        # cv.imshow('capa '+nom+" before",capa)
        resd = cv.morphologyEx(capa, cv.MORPH_CLOSE, kernel)
        # resd = cv.erode(cv.dilate(np.int16(capa),kernel,1),kernel,1)

        resd = np.array(resd, dtype=np.float)
        resultat = np.divide(capa,resd)
        # cv.imshow('capa '+nom+" after",resultat)
        # im[:,:,i] = resultat
        if(len(res)):
            res = res + (resultat,)
        else:
            res = (resultat,)
    im = np.dstack(res)
    print(type(im),)
    cv.imshow('neutralizada',im)
    cv.waitKey()

    return im

def fill_holes(im):
    _, contours, _ = cv.findContours(im,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv.drawContours(im,[cnt],0,255,-1)
    return im

def morf_method1(im):
    imggray = im.copy()
    # print(im.shape)
    # imggray = cv.cvtColor(im.copy(), cv.COLOR_RGB2GRAY)
    opening = imggray.copy()
    opening = cv.erode(opening,(2,2),iterations=1)
#     opening = cv.morphologyEx(opening, cv.MORPH_OPEN, small_kernel)
    # big_kernel = np.ones((6,6), np.uint8)
    biger_kernel = np.ones((8,8), np.uint8)
    bigerer_kernel = np.ones((14,14), np.uint8)
    bigerest_kernel = np.ones((20,20), np.uint8)

    # norm_kernel = np.ones((4,4), np.uint8)
    small_kernel = np.ones((3,3), np.uint8)
    # smaller_kernel = np.ones((2,2), np.uint8)
    # smallest_kernel = np.ones((1,1), np.uint8)
    h_kernel = np.ones((1,4), np.uint8)
    v_kernel = np.ones((4,1), np.uint8)

    opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, bigerest_kernel)
    opening = fill_holes(opening)
    opening = cv.morphologyEx(opening, cv.MORPH_OPEN, small_kernel)
    opening = cv.morphologyEx(opening, cv.MORPH_OPEN, h_kernel)
    opening = cv.morphologyEx(opening, cv.MORPH_OPEN, v_kernel)
    opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, bigerer_kernel)
    opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, bigerest_kernel)
    opening = fill_holes(opening)
    opening = cv.morphologyEx(opening, cv.MORPH_OPEN, biger_kernel)
    # opening = cv.erode(opening,bigerer_kernel,iterations=1)
    # opening = cv.morphologyEx(opening, cv.MORPH_CLOSE, bigerest_kernel)
    # opening = cv.dilate(opening,bigerest_kernel,iterations=1)
    imagen = fill_holes(opening)

    rows,cols= imagen.shape
    M = np.float32([[1,0,-5],[0,1,-5]])
    imagen = cv.warpAffine(imagen,M,(cols,rows))

    return imagen

def boundingBox_ccl(im):
    # im = cv.cvtColor(im.copy(), cv.COLOR_RGB2GRAY)
    _, contours, _ = cv.findContours(im,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    bb_list = list()
    for cnt in contours:
        x,y,w,h = cv.boundingRect(cnt)
        bb_list.append((x,y,w,h))
    return bb_list

def boundingBox_sw(im):
    # window with anchor on top left point
    bb_list = list()
    n, m = im.shape
    sw_size = 45 #args Dani needed
    step = 8
    for x in range(0, m-sw_size, step):
        for y in range(0, n-sw_size, step):
            #print(x,x+sw_size,y,y+sw_size) #The output coordinates are given as x1,x2,y1,y2
            window_img = im[y:y+sw_size,x:x+sw_size]
            fRatio = np.count_nonzero(window_img)/(sw_size*sw_size)
            if(fRatio > 0.5):
                bb_list.append((x,y,sw_size,sw_size))
    newbb = overlapped_windows(bb_list)
#    for x,y,w,h in newbb:
#        cv.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)

#    cv.imshow('sw', im)
#    cv.waitKey()
    return newbb

def is_intersect(x1,y1,w1,h1,x2,y2,w2,h2):
    is_intersecting = True
    if(x1 > x2+w2 or x1+w1 < x2):
        is_intersecting = False
    if(y1 > y2+h2 or y1+h1 < y2):
        is_intersecting = False
    return is_intersecting

def overlapped_windows(bb_list):
    # bb_overlaped = (bb,[related_bb_index])
    bb_overlapped = list()
    for x1,y1,w1,h1 in bb_list:
        related_bb_index = list()
        i=0
        for (x2,y2,w2,h2),_ in bb_overlapped:
            if is_intersect(x1,y1,w1,h1,x2,y2,w2,h2):
                related_bb_index.append(i)
            i +=1
        bb_overlapped.append(((x1,y1,w1,h1),related_bb_index))

    corespondence_dict = dict()
    bb_cell = list()
    i = 0
    for bb,index_list in bb_overlapped:
        if(len(index_list) > 0):
            bb_cell[corespondence_dict[index_list[0]]].append(bb)
            corespondence_dict[i] = corespondence_dict[index_list[0]]
        else:
            bb_cell.append([bb])
            corespondence_dict[i] = len(bb_cell)-1 #esto habria que hacerlo para todas las posiciones de la lista
        i += 1

    final_bbs = list()
    for group in bb_cell:
        x, y, x2, y2 = float('inf'),float('inf'),0,0
        for bb in group:
            x = min(bb[0],x)
            y = min(bb[1],y)
            x2 = max(bb[0]+bb[2], x2)
            y2 = max(bb[1]+bb[3], y2)
        final_bbs.append((x,y,x2-x,y2-y))

    return final_bbs




def boundingBoxFilter_method1(im, bb_list):
    image = im.copy()
    pixels = []
    for x,y,w,h in bb_list:
        f_ratio = np.sum(image[y:y+h, x:x+w] > 0)/float(w*h)
        form_factor = float(w)/h
        if(w*h < 700 or w*h > 20000 or f_ratio < 0.3 or form_factor < 0.333 or form_factor > 3):
            image[y:y+h, x:x+w] = np.zeros((h,w))

    return image



# Create your own candidate_generation_pixel_xxx functions for other color spaces/methods
# Add them to the switcher dictionary in the switch_methods() function
# These functions should take an image as input and output the pixel_candidates mask image


def switch_methods(im):
    """
    Performs pixel generator whith method selection
    * Inputs:
    - im = skimage.io image to be analized
    - color_space = method selector (switcher variable below)
    - preprocess_treatment = preprocess filter/function to be applied before candidate
                             selection (multiple preprocess available)
    *Outputs:
    - pixel_candidates: mask of pixels candidates for possible signals
    """
    from main import CONSOLE_ARGUMENTS

    switcher = {
        'rgb': candidate_generation_pixel_rgb,
        'luv'    : candidate_generation_pixel_luv,
        'hsv'	 : candidate_generation_pixel_hsv_team1,
        'hsv-rgb': candidate_generation_pixel_hsvb_rgbr,
        'lab'    : candidate_generation_pixel_lab,
        'luv-rgb' : candidate_generation_pixel_luvb_rgbr,
        'GW-luv-rgb': candidate_generation_GW_pixel_luv,
        'luv-hsv' : candidate_generation_pixel_luvb_hsvr,
        'normRGB-luv-rgb' : candidate_generation_pixel_normrgb_luvb_rgbr
    }

    switcher_preprocess = {
        'blur': preprocess_blur,
        #'normrgb' : preprocess_normrgb,
        'whitePatch': preprocess_whitePatch,
        'grayWorld' : preprocess_grayWorld,
        'neutralize': preprocess_neutre
    }

    switcher_morf = {
        'm1': morf_method1
    }

    switcher_bb = {
        'ccl': boundingBox_ccl,
        'sw': boundingBox_sw
    }

    switcher_window = {
        'm1': boundingBoxFilter_method1
    }

    # print(CONSOLE_ARGUMENTS.prep_pixel_selector)
    pixel_selector = CONSOLE_ARGUMENTS.pixel_selector
    preprocess = CONSOLE_ARGUMENTS.prep_pixel_selector
    morphology = CONSOLE_ARGUMENTS.morphology
    boundingBox = CONSOLE_ARGUMENTS.boundingBox
    window = CONSOLE_ARGUMENTS.window
    reduce_bbs = CONSOLE_ARGUMENTS.reduce_bbs
    view_img = CONSOLE_ARGUMENTS.view_imgs
    
    # PIXEL PREPROCESS
    if preprocess is not None:
        if not isinstance(preprocess, list):
            preprocess = list(preprocess)
        for preproc in preprocess:
            func = switcher_preprocess.get(preproc, lambda: "Invalid preprocess")
            im = func(im)

    # PIXEL SELECTOR
    func = switcher.get(pixel_selector, lambda: "Invalid color segmentation method")
    pixel_candidates = func(im)
    pixel_candidates = pixel_candidates.astype('uint8')
    # print("\nPIX:", pixel_candidates.shape)
    # PIXEL MORPHOLOGY
    if morphology is not None:
        if not isinstance(morphology, list):
            morphology = list(morphology)
        for preproc in morphology:
            func = switcher_morf.get(preproc, lambda: "Invalid morphology")
            pixel_candidates = func(pixel_candidates)

    bb_list = None
            
    # PIXEL BB
    if boundingBox is not None:
        if not isinstance(boundingBox, list):
            boundingBox = list(boundingBox)
        for preproc in boundingBox:
            func = switcher_bb.get(preproc, lambda: "Invalid bounding box")
            bb_list = func(pixel_candidates)
            if(reduce_bbs): bb_list = reduce_winds_sizes(bb_list, pixel_candidates)
            


    # PIXEL WINDOW
    if window is not None and bb_list is not None:
        if not isinstance(window, list):
            window = list(window)
        for preproc in window:
            func = switcher_window.get(preproc, lambda: "Invalid window")
            pixel_candidates = func(pixel_candidates, bb_list)

    return pixel_candidates

def candidate_generation_pixel(im):
    pixel_candidates = switch_methods(im)
    pixel_candidates = pixel_candidates.astype('uint8')
    # pixel_candidates = remove_small_noise(pixel_candidates)
    msk = np.dstack([pixel_candidates]*3)
    # immask = msk*im
    # cv.imshow("test.png",immask)
    # cv.imshow("imageb",im)
    # cv.waitKey(0)

    return pixel_candidates

def remove_small_noise(im):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(500,500))
    cv.morphologyEx(im, cv.MORPH_BLACKHAT, kernel)
    return im
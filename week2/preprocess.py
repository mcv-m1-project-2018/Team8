# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np

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

def preprocess_image(im, preprocesses):
    switcher_preprocess = {
        'blur': preprocess_blur,
        #'normrgb' : preprocess_normrgb,
        'whitePatch': preprocess_whitePatch,
        'grayWorld' : preprocess_grayWorld,
        'neutralize': preprocess_neutre
    }
    
    # PIXEL PREPROCESS
    if preprocesses is not None:
        if not isinstance(preprocesses, list):
            preprocesses = list(preprocesses)
        for preprocess in preprocesses:
            func = switcher_preprocess.get(preprocess, lambda: "Invalid preprocess")
            im = func(im)
    return im
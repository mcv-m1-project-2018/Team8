#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from skimage import color
import cv2 as cv

from candidate_generation_window import reduce_winds_sizes
from preprocess import preprocess_normrgb

def masks_rgb(im, rest=0):
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
    mskr = mskr*(image[:,:,1] < 50-rest)
    mskr = mskr*(image[:,:,2] < 50+rest)

    #blue colored signals
    mskb = image[:,:,0] < 50
    mskb = mskb*(image[:,:,1] < 100-rest)
    mskb = mskb*(image[:,:,2] > 60+rest)

    return mskr, mskb


def mask_luv(im, rest=0):
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
    mskb = mskb*(image[:,:,2] < 114-rest)

    mskr = image[:,:,2] > 127
    mskr = mskr*(image[:,:,2] < 157-rest)
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

def mask_hsv(im, rect_down=0, rect_up=0):
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

    mask_red = (((hsv_im[:,:,0] < 0.027+rect_down) | (hsv_im[:,:,0] > 0.93+rect_up)))
    mask_red = mask_red & (hsv_im[:,:,1] > 0.27) & (hsv_im[:,:,1] < 0.95)
    mask_red = mask_red & (hsv_im[:,:,2] > 0.19) & (hsv_im[:,:,2] < 0.90)
    
    mask_blue = ((hsv_im[:,:,0] > 0.55+rect_down) & (hsv_im[:,:,0] < 0.75+rect_up))


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
    mskr, _ = masks_rgb(im, +15)
    _ , mskb = mask_luv(im, -15)
    return mskb+mskr

def candidate_generation_pixel_luvb_rgbr2(im): 
    mskr, _ = masks_rgb(im, -10)
    _ , mskb = mask_luv(im, -10)
    return mskb+mskr
def candidate_generation_pixel_luvb_rgbr3(im): 
    mskr, _ = masks_rgb(im, -5)
    _ , mskb = mask_luv(im, -5)
    return mskb+mskr
def candidate_generation_pixel_luvb_rgbr4(im): 
    mskr, _ = masks_rgb(im, 0)
    _ , mskb = mask_luv(im, 0)
    return mskb+mskr

def candidate_generation_pixel_luvb_hsvr(im):
    mskr, _ = mask_hsv(im)
    _ , mskb = candidate_generation_pixel_luv(im)
    return mskb+mskr

def candidate_generation_pixel_luvb_hsvr2(im): 
    mskr, _ = mask_hsv(im, -0.00, +0.06)
    _ , mskb = mask_luv(im, -15)
    return mskb+mskr
def candidate_generation_pixel_normrgb_luvb_rgbr(im):
    noiseMask = candidate_generation_pixel_normrgb(im)
    msk = candidate_generation_pixel_luvb_rgbr(im)
    msk = msk*(np.logical_not(noiseMask))
    return msk

###############################


# Create your own candidate_generation_pixel_xxx functions for other color spaces/methods
# Add them to the switcher dictionary in the switch_methods() function
# These functions should take an image as input and output the pixel_candidates mask image


def switch_methods(im, pixel_selector):
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
    
    switcher = {
        'rgb': candidate_generation_pixel_rgb,
        'luv'    : candidate_generation_pixel_luv,
        'hsv'	 : candidate_generation_pixel_hsv_team1,
        'hsv-rgb': candidate_generation_pixel_hsvb_rgbr,
        'lab'    : candidate_generation_pixel_lab,
        'luv-rgb' : candidate_generation_pixel_luvb_rgbr,
        'luv-rgb2' : candidate_generation_pixel_luvb_rgbr2,
        'luv-rgb3' : candidate_generation_pixel_luvb_rgbr3,
        'luv-rgb4' : candidate_generation_pixel_luvb_rgbr4,
        'GW-luv-rgb': candidate_generation_GW_pixel_luv,
        'luv-hsv' : candidate_generation_pixel_luvb_hsvr,
        'luv-hsv2' : candidate_generation_pixel_luvb_hsvr2,
        'normRGB-luv-rgb' : candidate_generation_pixel_normrgb_luvb_rgbr
    }

    # print(CONSOLE_ARGUMENTS.prep_pixel_selector)

    

    # PIXEL SELECTOR
    func = switcher.get(pixel_selector, lambda: "Invalid color segmentation method")
    pixel_candidates = func(im)
    pixel_candidates = pixel_candidates.astype('uint8')
    # print("\nPIX:", pixel_candidates.shape)
    return pixel_candidates


def candidate_generation_pixel(im, pixel_selector):
    pixel_candidates = switch_methods(im, pixel_selector)
    return pixel_candidates

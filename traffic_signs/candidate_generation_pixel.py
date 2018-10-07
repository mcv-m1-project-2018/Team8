#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from skimage import color
import cv2 as cv


def masks_rgb(im):

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
	image = im[:,:,:]
	image = cv.cvtColor(image,cv.COLOR_RGB2Luv)

	mskb = image[:,:,2] > 68
	mskb = mskb*(image[:,:,2] < 114)

	mskr = image[:,:,2] > 127
	mskr = mskr*(image[:,:,2] < 157)
	return mskr, mskb

def mask_lab(im):
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

#############################################
def candidate_generation_pixel_rgb(im):
	mskr, mskb = masks_rgb(im)
	return mskr+mskb

def candidate_generation_pixel_hsv(im):
	# convert input image to HSV color space
	hsv_im = color.rgb2hsv(im)
	
	# Develop your method here:
	# Example:
	pixel_candidates = hsv_im[:,:,1] > 0.4

	return pixel_candidates

def candidate_generation_pixel_lab(im):
	mskr, mskb = mask_lab(im)
	return mskr+mskb

def candidate_generation_pixel_luv(im):
	mskr, mskb = mask_luv(im)
	return mskr, mskb 

def candidate_generation_pixel_normrgb(im): 
	im = preprocess_normrgb(im)

	# filter to get noise:
	mskr = im[:,:,0] > 20
	mskr = mskr*(im[:,:,1] > 20)
	mskr = mskr*(im[:,:,2] > 20)

	return mskr

def candidate_generation_pixel_gw_rgb(im): 
	return candidate_generation_pixel_rgb(preprocess_grayWorld(im))

def candidate_generation_pixel_luvb_rgbr(im): 

	mskr, _ = masks_rgb(im)

	_ , mskb = candidate_generation_pixel_luv(im)

	return mskb+mskr

def candidate_generation_pixel_wp_rgb(im): 
	return candidate_generation_pixel_rgb(preprocess_grayWorld(im))

def candidate_generation_pixel_blur_luvb_rgbr(im):
	return candidate_generation_pixel_luvb_rgbr(preprocess_blur(im))

def candidate_generation_pixel_gw_blur_luvb_rgbr(im):
	return candidate_generation_pixel_luvb_rgbr(preprocess_blur(preprocess_grayWorld(im)))

###############################
def preprocess_blur(im):
	window_mean = 5
	blurred_img = cv.blur(im,(window_mean, window_mean))
	return blurred_img

def preprocess_normrgb(im):
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
	Funcio que neutralitza el color de la imatge. Util quan es esta tacat o hi
	ha variancies.
	* Inputs:
	- im = skimage.io image
	*Outputs:
	- im = imatge amb color neutralitzat
	"""
	
	[x, y] = im.shape
	sz = np.nonzero(x/85)
	if (sz < 5):
		sz = 5
	kernel = np.ones((sz,sz),np.uint8)
	resd = cv.erode(cv.dilate(np.int16(im),kernel,1),kernel,1)

	resd = np.array(resd, dtype=np.float)
	im = np.divide(im,resd)
	return im

# Create your own candidate_generation_pixel_xxx functions for other color spaces/methods
# Add them to the switcher dictionary in the switch_methods() function
# These functions should take an image as input and output the pixel_candidates mask image


def switch_methods(im, color_space, preprocess=None):
	from main import CONSOLE_ARGUMENTS

	switcher = {
		'rgb': candidate_generation_pixel_rgb,
		'luv'    : candidate_generation_pixel_luv,
		'lab'    : candidate_generation_pixel_lab,
		'luv-rgb' : candidate_generation_pixel_luvb_rgbr,
		'Blur-luv-rgb' : candidate_generation_pixel_blur_luvb_rgbr,
		'GW-Blur-luv-rgb' :candidate_generation_pixel_gw_blur_luvb_rgbr,
		'GW-RGB'    : candidate_generation_pixel_gw_rgb,
		'WP-RGB'    : candidate_generation_pixel_wp_rgb
	}

	switcher_preprocess = {
		'blur': preprocess_blur,
		'normrgb' : preprocess_normrgb,
		'whitePatch': preprocess_whitePatch,
		'grayWorld' : preprocess_grayWorld,
		'neutralize': preprocess_neutre
	}

	print(CONSOLE_ARGUMENTS.prep_pixel_selector)
	preprocess = CONSOLE_ARGUMENTS.prep_pixel_selector

	if preprocess is not None:
		if not isinstance(preprocess, list):
			preprocess = list(preprocess)
		for preproc in preprocess:
			im = switcher_preprocess[preproc](im)
	
	# Get the function from switcher dictionary
	func = switcher.get(color_space, lambda: "Invalid color space")

	# Execute the function
	pixel_candidates = func(im)

	return pixel_candidates

def candidate_generation_pixel(im, color_space):
	pixel_candidates = switch_methods(im, color_space)
	# msk = np.dstack([pixel_candidates]*3)
	# immask = msk*im
	# cv.imshow("test.png",immask)
	# cv.imshow("imageb",im)
	# cv.waitKey(0)

	return pixel_candidates
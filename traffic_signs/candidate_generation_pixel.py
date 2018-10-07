#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from skimage import color
import cv2 as cv

def candidate_generation_pixel_rgb(im):
	
	# Develop your method here:
	# Example:
	mskr = im[:,:,0] > 70
	mskr = mskr*(im[:,:,1] < 50)
	mskr = mskr*(im[:,:,2] < 50)

	#blue colored signals
	mskb = im[:,:,0] < 50
	mskb = mskb*(im[:,:,1] < 100)
	mskb = mskb*(im[:,:,2] > 60)

	msk = mskr + mskb

	pixel_candidates = msk

	return pixel_candidates
 
def candidate_generation_pixel_hsv(im):
	# convert input image to HSV color space
	hsv_im = color.rgb2hsv(im)
	
	# Develop your method here:
	# Example:
	pixel_candidates = hsv_im[:,:,1] > 0.4

	return pixel_candidates

def candidate_generation_pixel_lab(im):
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

	msk = mskr + mskb

	return msk

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

# Create your own candidate_generation_pixel_xxx functions for other color spaces/methods
# Add them to the switcher dictionary in the switch_methods() function
# These functions should take an image as input and output the pixel_candidates mask image
def candidate_generation_pixel_gw_rgb(im): 
	return candidate_generation_pixel_rgb(preprocess_grayWorld(im))

def candidate_generation_pixel_wp_rgb(im): 
	return candidate_generation_pixel_rgb(preprocess_grayWorld(im))

def candidate_generation_pixel_blur_rgb(im): 
	return candidate_generation_pixel_rgb(preprocess_blur(im))
	
def candidate_generation_pixel_gw_blur_rgb(im): 
	return candidate_generation_pixel_rgb(preprocess_blur(preprocess_grayWorld(im)))


def switch_methods(im, color_space):
	switcher = {
		'rgb': candidate_generation_pixel_rgb,
		'hsv'    : candidate_generation_pixel_hsv,
		'lab'    : candidate_generation_pixel_lab,
		'GW-RGB'    : candidate_generation_pixel_gw_rgb,
		'WP-RGB'    : candidate_generation_pixel_wp_rgb,
		'Blur-RGB'    : candidate_generation_pixel_blur_rgb,
		'GW-Blur-RGB'    : candidate_generation_pixel_gw_blur_rgb
	}

	# Get the function from switcher dictionary
	func = switcher.get(color_space, lambda: "Invalid color space")

	# Execute the function
	pixel_candidates =  func(im)

	return pixel_candidates


def candidate_generation_pixel(im, color_space):

	pixel_candidates = switch_methods(im, color_space)
	# msk = np.dstack([pixel_candidates]*3)
	# immask = msk*im
	# cv.imshow("asd",im)
	# cv.imshow("asd",immask)
	# cv.waitKey(0)

	return pixel_candidates
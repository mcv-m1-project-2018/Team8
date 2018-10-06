
import imageio
import fnmatch
import os
import sys
import cv2 as cv
from evaluation.load_annotations import load_annotations
import numpy as np
from evaluation.evaluation_funcs import performance_accumulation_pixel,performance_evaluation_pixel

def whitePatch(im):
	bmax, gmax, rmax = np.amax(np.amax(im,axis=0),axis=0)

	alpha = gmax/rmax
	beta = gmax/bmax

	im[:,:,2] = alpha*im[:,:,2]
	im[:,:,0] = beta*im[:,:,0]

	return im

def grayWorld(im):
	bmean, gmean, rmean = np.mean(np.mean(im,axis=0),axis=0)

	alpha = gmean/rmean
	beta = gmean/bmean

	im[:,:,2] = alpha*im[:,:,2]
	im[:,:,0] = beta*im[:,:,0]

	return im

def RGBMaskFilters(im):
	#red colored signals
	mskr = im[:,:,2] > 70
	mskr = mskr*(im[:,:,1] < 50)
	mskr = mskr*(im[:,:,0] < 50)
	mskr = np.dstack([mskr]*3)

	#blue colored signals
	mskb = im[:,:,2] < 50
	mskb = mskb*(im[:,:,1] < 100)
	mskb = mskb*(im[:,:,0] > 60)
	mskb = np.dstack([mskb]*3)

	msk = mskr + mskb

	return msk

# def HSVFilter(im):
# 	imagen = im[:,:,:]

# 	imagen = cv.cvtColor(imagen,cv.COLOR_BGR2HSV)

# 	msk = imagen[:,:,0] < 21
# 	msk = msk+(imagen[:,:,0] > 210)
# 	msk = msk*(imagen[:,:,1] > 120)
# 	msk = msk*(imagen[:,:,2] > 70)
# 	msk = np.dstack([msk]*3)

# 	return msk

def LabFilter(im):
	imagen = im[:,:,:]

	imagen = cv.cvtColor(imagen,cv.COLOR_BGR2Lab)

	mskb = imagen[:,:,2] < 115
	mskb = mskb*(imagen[:,:,0] > 40)
	mskb = mskb*(imagen[:,:,1] < 200)
	mskb = mskb*(imagen[:,:,2] > 35)
	mskb = np.dstack([mskb]*3)

	mskr = imagen[:,:,1] > 140
	mskr = mskr*(imagen[:,:,0] > 20)
	mskr = mskr*(imagen[:,:,0] < 220)
	mskr = mskr*(imagen[:,:,2] < 150)
	mskr = mskr*(imagen[:,:,2] > 125)
	mskr = np.dstack([mskr]*3)

	msk = mskr + mskb

	return msk


def segmentate(im_directory, mask_directory,maskOut_directory):
	file_names = sorted(fnmatch.filter(os.listdir(im_directory), '*.jpg'))

	pixelTP = 0
	pixelFP = 0
	pixelFN = 0
	pixelTN = 0

	#For each file 
	for name in file_names[:10]:
		base, extension = os.path.splitext(name)
		
		imageNameFile = im_directory + "/" + name
		maskNameFile = mask_directory + "/mask." + base + ".png"

		print(imageNameFile)
		image = cv.imread(imageNameFile)
		maskImage = cv.imread(maskNameFile)

		msk = LabFilter(image)

		img = image*msk
		msk = msk.astype(float)

		# cv.imshow("masked image",msk)
		# cv.imshow("original image",image)
		# cv.waitKey(0)
		
		# cv.imwrite(os.path.join(maskOut_directory,("mask." + base + ".png")),msk)
		pTP, pFP, pFN, pTN = performance_accumulation_pixel(msk,maskImage)
		pixelTP += pTP
		pixelFP += pFP
		pixelFN += pFN
		pixelTN += pTN

	print("precision \t accuracy \t specificity \t sensitivity")
	print(performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN))


def main():
	im_directory = "./Dataset/train"
	mask_directory = "./Dataset/train/mask/"
	maskOut_directory = "./Dataset/maskOut/"

	segmentate(im_directory,mask_directory,maskOut_directory)

if __name__ == '__main__':
	main()
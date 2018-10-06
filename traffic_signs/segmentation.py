
import imageio
import fnmatch
import os
import sys
import cv2 as cv
from evaluation.load_annotations import load_annotations
import numpy as np

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

def handPickedMaskFilters(im):
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


def segmentate(im_directory, mask_directory):
	file_names = sorted(fnmatch.filter(os.listdir(im_directory), '*.jpg'))

	#For each file 
	for name in file_names:
		base, extension = os.path.splitext(name)
		
		imageNameFile = im_directory + "/" + name
		print(imageNameFile)
		image = cv.imread(imageNameFile)
		image = grayWorld(image)

		msk = handPickedMaskFilters(image)

		img = image*msk
		cv.imshow("masked image",img)
		cv.imshow("original image",image)
		cv.waitKey(0)

		msk = msk.astype(int)
		cv.imwrite(os.path.join(mask_directory,("mask." + base + ".png")),msk)


def main():
	im_directory = "./Dataset/train"
	mask_directory = "./Dataset/maskOut"

	segmentate(im_directory,mask_directory)

if __name__ == '__main__':
	main()
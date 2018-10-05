
import imageio
import fnmatch
import os
import sys
import cv2 as cv
from evaluation.load_annotations import load_annotations
import numpy as np

def whitePatch(im):
	bmax, gmax, rmax = np.max(im,axis=2)

	print(bmax,gmax,rmax)

def segmentate(im_directory):
	file_names = sorted(fnmatch.filter(os.listdir(im_directory), '*.jpg'))


	#For each file 
	for name in file_names:
		base, extension = os.path.splitext(name)

		
		imageNameFile = im_directory + "/" + name
		print(imageNameFile)
		image = cv.imread(imageNameFile)

		#red filter
		mskr = image[:,:,2] > 70
		mskr = mskr*(image[:,:,1] < 50)
		mskr = mskr*(image[:,:,0] < 50)
		mskr = np.dstack([mskr]*3)

		mskb = image[:,:,2] < 50
		mskb = mskb*(image[:,:,1] < 100)
		mskb = mskb*(image[:,:,0] > 60)
		mskb = np.dstack([mskb]*3)
		msk = mskr + mskb


		img = image*msk

		cv.imshow("asd",img)
		cv.imshow("das",image)
		cv.waitKey(0)


def main():
	im_directory = "./Dataset/train"

	segmentate(im_directory)

if __name__ == '__main__':
	main()
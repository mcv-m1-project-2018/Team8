import imageio
import fnmatch
import os
import sys
import cv2 as cv
from evaluation.load_annotations import load_annotations
import numpy as np

def showImageAnnotationsAndMask(image, mask, annotations):
	imageRects = np.copy(image)
	for rect in annotations:
		cv.rectangle(imageRects,(int(rect[1]),int(rect[0])),(int(rect[3]),int(rect[2])),(0,0,255),2)
	cv.imshow("Image",imageRects)
	cv.imshow("Mask",imageMask*255)

	cv.waitKey(0)

#Start declaring variables for signal analysis
signal_dicts = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}

directory = "./Dataset/train"
mask_directory = "./Dataset/train/mask"
groundtruth_directory = "./Dataset/train/gt"

file_names = sorted(fnmatch.filter(os.listdir(directory), '*.jpg'))

#database status variables
maxList = [0]*len(signal_dicts)
minList = [999999999]*len(signal_dicts)
formFactorList = l = [[] for _ in range(len(signal_dicts))]
fillingRatioList = [[] for _ in range(len(signal_dicts))]
freqApparition = [0]*len(signal_dicts)

#For each file we extract and fill previous database status variables
print("tly\t tlx\t bly\t blx\t ones\t total\t fratio")
for name in file_names[:10]:
	base, extension = os.path.splitext(name)

	imageNameFile = directory + "/" + name
	maskNameFile = mask_directory + "/mask." + base + ".png"
	gtNameFile = groundtruth_directory + "/gt." + base + ".txt"

	image = cv.imread(imageNameFile)
	imageMask = cv.imread(maskNameFile)
	annotations = load_annotations(gtNameFile)

	

	# showImageAnnotationsAndMask(image,imageMask,annotations)
	# print(annotations)
	for rect in annotations:
		tly, tlx, bly, blx = rect[0:3]
		signal_type = signal_dicts[rect[4]]

		#form factor calculations
		width = abs(blx-tlx)
		height = abs(tly-bly)
		formFactorList[signal_type].append(width/height)

		#filling and max and min calculations
		ones = np.count_nonzero(imageMask[tly:bly,tlx:blx,0])
		total= width*height
		fratio = ones/float(total)
		maxList[signal_type] = max(maxList[signal_type],ones)
		minList[signal_type] = min(minList[signal_type],ones)
		fillingRatioList[signal_type].append(fratio)
		
		print( tly,"\t", tlx,"\t", bly,"\t", blx,"\t {0:.0f}".format(ones),"\t{0:.0f}".format(total),"\t{0:.4f}".format(fratio))
		#Frequency
		freqApparition[signal_type] += 1
	
print()
print(fillingRatioList)
print(maxList)
print(minList)
print(formFactorList)

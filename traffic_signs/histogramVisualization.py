

import imageio
import fnmatch
import os
import sys
import cv2 as cv
from evaluation.load_annotations import load_annotations
import numpy as np
from evaluation.evaluation_funcs import performance_accumulation_pixel,performance_evaluation_pixel
from matplotlib import pyplot as plt
from candidate_generation_pixel import preprocess_normrgb


signal_dicts = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}

def visualizeHistograms(imPath, gtPath, maskPath, colorSpace = "RGB"):
	file_names = sorted(fnmatch.filter(os.listdir(imPath), '*.jpg'))

	histAll = [[[0] for _ in range(3)] for _ in range(6)]
	for name in file_names[:-1]:
		base, extension = os.path.splitext(name)
		
		imageNameFile = imPath + "/" + name
		maskNameFile = maskPath + "/mask." + base + ".png"
		gtNameFile = gtPath + "/gt." + base + ".txt"

		image = cv.imread(imageNameFile)

		image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
		image = preprocess_normrgb(image)
		image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
		if(colorSpace == "LAB"):
			image = cv.cvtColor(image,cv.COLOR_BGR2Lab)
		if(colorSpace == "Luv"):
			image = cv.cvtColor(image,cv.COLOR_BGR2Luv)
		if(colorSpace == "normRGB"):
			image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
			image = preprocess_normrgb(image)
		elif(colorSpace == "HSL" ):
			image = cv.cvtColor(image,cv.COLOR_BGR2HLS)
		elif(colorSpace == "HSV" ):
			image = cv.cvtColor(image,cv.COLOR_BGR2HSV)
		elif(colorSpace == "Yuv" ):
			image = cv.cvtColor(image,cv.COLOR_BGR2YUV)
		elif(colorSpace == "XYZ" ):
			image = cv.cvtColor(image,cv.COLOR_BGR2XYZ)
		elif(colorSpace == "YCrCb" ):
			image = cv.cvtColor(image,cv.COLOR_BGR2YCrCb)

		maskImage = cv.imread(maskNameFile)
		image = image * maskImage
		annot = load_annotations(gtNameFile)

		for rect in annot:
			tly, tlx, bly, blx = rect[:4]
			tly, tlx, bly, blx = int(tly), int(tlx), int(bly), int(blx)

			color = ('b','g','r')
			for i in range(3):
				histr = cv.calcHist(image[tly:bly,tlx:blx,:],[i],None,[255],[0,256])
				histr[0] = 0
				histAll[ signal_dicts[rect[4]]][i] += histr
				

	titles=["A","B","C","D","E","F"]
	for j, hist_signal_type in enumerate(histAll):
		color = ('b','g','r')
		plt.figure()
		plt.title(titles[j])
		plt.ioff()
		for i, col in enumerate(color):
			hist_signal_type[i][0] = 0
			plt.plot(hist_signal_type[i],color = col)
			plt.xlim([0,255])
		
		directory = "./Dataset/histogramNormPrecise/"+colorSpace
		if not os.path.exists(directory):
			os.makedirs(directory)
		plt.savefig(directory+"/norm_"+titles[j]+".png")


def main():
	im_directory = "./Dataset/train"
	mask_directory = "./Dataset/train/mask"
	gt_directory = "./Dataset/train/gt"

	colorSpaces = ["Luv"]
	for color in colorSpaces:
		print(color)
		visualizeHistograms(im_directory,gt_directory,mask_directory,color)

if __name__ == '__main__':
	main()

import imageio
import fnmatch
import os
import sys
import cv2 as cv
from evaluation.load_annotations import load_annotations
import numpy as np
from evaluation.evaluation_funcs import performance_accumulation_pixel,performance_evaluation_pixel
from matplotlib import pyplot as plt

signal_dicts = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}

def whitePatch(im):
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

def grayWorld(im):
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

def HSLFilter(im):
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


def trainModel(imPath, gtPath, maskPath):
	file_names = sorted(fnmatch.filter(os.listdir(imPath), '*.jpg'))

	histAll = [[[0] for _ in range(3)] for _ in range(6)]
	for name in file_names[:3]:
		base, extension = os.path.splitext(name)
		
		imageNameFile = imPath + "/" + name
		maskNameFile = maskPath + "/mask." + base + ".png"
		gtNameFile = gtPath + "/gt." + base + ".txt"

		# print(imageNameFile)
		image = cv.imread(imageNameFile)
		# image = cv.cvtColor(image,cv.COLOR_BGR2HSV)
		image = cv.cvtColor(image,cv.COLOR_BGR2HLS)
		plt.figure()
		ax1 = plt.subplot(131)
		ax2 = plt.subplot(132)
		ax3 = plt.subplot(133)
		ax1.imshow(image[:,:,0])
		ax2.imshow(image[:,:,1])
		ax3.imshow(image[:,:,2])
		plt.show()
		maskImage = cv.imread(maskNameFile)
		image = image * maskImage
		annot = load_annotations(gtNameFile)

		for rect in annot:
			tly, tlx, bly, blx = rect[:4]
			tly, tlx, bly, blx = int(tly), int(tlx), int(bly), int(blx)

			color = ('b','g','r')
			# vals = image[tly:bly,tlx:blx]
			# plt.hist(vals[0][vals[0] != 0], histtype="step",color="blue")
			# plt.hist(vals[1][vals[1] != 0], histtype="step",color="green")
			# plt.hist(vals[2][vals[2] != 0], histtype="step",color="red")
			# plt.figure()
			# plt.show()
			for i in range(3):
				histr = cv.calcHist(image[tly:bly,tlx:blx,:],[i],None,[60],[0,256])
				print("Píxels: ", len(image[tly:bly,tlx:blx,i]))
				histr[0] = 0
				histAll[ signal_dicts[rect[4]]][i] += histr
				
				# plt.plot(histr,color = color[i])
				# plt.xlim([0,60])
			print("-------")
			# cv.imshow("asda",image[tly:bly,tlx:blx,:])
			# plt.show()
			# cv.waitKey()


	print("histograting")
	titles=["A","B","C","D","E","F"]
	for j, hist_signal_type in enumerate(histAll):
		color = ('b','g','r')
		plt.figure()
		plt.title(titles[j])
		plt.ioff()
		for i, col in enumerate(color):
			hist_signal_type[i][0] = 0
			plt.plot(hist_signal_type[i],color = col)
			plt.xlim([0,60])
		plt.savefig("C:/Users/richa/Desktop/"+titles[j]+".png")
	# cv.waitKey()



		

def segmentate(im_directory, mask_directory,maskOut_directory):
	file_names = sorted(fnmatch.filter(os.listdir(im_directory), '*.jpg'))

	pixelTP = 0
	pixelFP = 0
	pixelFN = 0
	pixelTN = 0

	#For each file 
	for name in file_names[:100]:
		base, extension = os.path.splitext(name)
		
		imageNameFile = im_directory + "/" + name
		maskNameFile = mask_directory + "/mask." + base + ".png"

		print(imageNameFile)
		image = cv.imread(imageNameFile)
		maskImage = cv.imread(maskNameFile)

		
		# image = grayWorld(image)
		msk = LabFilter(image)


		img = image*msk
		msk = msk.astype(float)

		cv.imshow("masked image",msk)
		cv.imshow("original image",image)
		cv.waitKey(0)
		
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
	mask_directory = "./Dataset/train/mask"
	gt_directory = "./Dataset/train/gt"
	maskOut_directory = "./Dataset/maskOut"

	# segmentate(im_directory,mask_directory,maskOut_directory)
	trainModel(im_directory,gt_directory,mask_directory)

if __name__ == '__main__':
	main()
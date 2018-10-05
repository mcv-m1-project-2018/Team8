import imageio
import fnmatch
import os
import sys
import cv2 as cv
from evaluation.load_annotations import load_annotations
import numpy as np

#Start declaring variables for signal analysis
signal_dicts = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5}

class Signal(object):
	def __init__(self):
		#Images
		self.img_orig = None # Original image of the signal (training)
		self.img_mask = None # Mask of the signal (training)
		self.img_pred = None # Predicted signal (Test)
		
		#It's position in original images
		self.tly = None
		self.tlx = None
		self.bly = None
		self.blx = None

		#Characteristics
		self.signal_type = None
		self.width = None
		self.height = None
		self.pixels = None
		self.fratio = None # Filling Ratio
		self.ffactor= None # Form Factor

		

	def build_from_annotation(self, annotation, image, imageMask, saveImg=False):
		tly, tlx, bly, blx = annotation[:4]
		tly, tlx, bly, blx = int(tly), int(tlx), int(bly), int(blx)
		self.tly, self.tlx, self.bly, self.blx = tly, tlx, bly, blx

		self.signal_type = annotation[4]
		self.width = abs(blx-tlx)
		self.height = abs(tly-bly)
		self.ffactor = self.width/self.height

		img_orig = image[tly:bly,tlx:blx,0]
		img_mask = imageMask[tly:bly,tlx:blx,0]

		ones = np.count_nonzero(img_mask)
		total= self.width*self.height
		self.pixels = total
		self.fratio = ones/float(total)

		if(saveImg):
			self.img_orig = img_orig
			self.img_mask = img_mask

# def extract_signal_from_annotation(rect, image, imageMask):


	#filling and max and min calculations

	
	# maxList[signal_type] = max(maxList[signal_type],ones)
	# minList[signal_type] = min(minList[signal_type],ones)
	# fillingRatioList[signal_type].append(fratio)
	
	# print( tly,"\t", tlx,"\t", bly,"\t", blx,"\t {0:.0f}".format(ones),"\t{0:.0f}".format(total),"\t{0:.4f}".format(fratio))
	#Frequency
	# freqApparition[signal_type] += 1

def extract_signals_im_training(imageNameFile, maskNameFile, gtNameFile):
	image = cv.imread(imageNameFile)
	imageMask = cv.imread(maskNameFile)
	annotations = load_annotations(gtNameFile)

	# showImageAnnotationsAndMask(image,imageMask,annotations)
	# print(annotations)
	signal_list = []
	for annotation in annotations:
		signal = Signal()
		signal.build_from_annotation(annotation, image, imageMask)
		signal_list.append(signal)
	return signal_list

# def showImageAnnotationsAndMask(image, mask, annotations):
# 	imageRects = np.copy(image)
# 	for rect in annotations:
# 		cv.rectangle(imageRects,(int(rect[1]),int(rect[0])),(int(rect[3]),int(rect[2])),(0,0,255),2)
# 	cv.imshow("Image",imageRects)
# 	cv.imshow("Mask",mask*255)

# 	cv.waitKey(0)

def calculateImagesMetrics(im_directory,mask_directory,gt_directory, files_to_process=-1):
	file_names = sorted(fnmatch.filter(os.listdir(im_directory), '*.jpg'))

	#database status variables
	# maxList = [0]*len(signal_dicts)
	# minList = [float('inf')]*len(signal_dicts)
	# formFactorList = [[] for _ in range(len(signal_dicts))]
	# fillingRatioList = [[] for _ in range(len(signal_dicts))]
	# freqApparition = [0]*len(signal_dicts)

	#For each file we extract and fill previous database status variables
	# print("tly\t tlx\t bly\t blx\t ones\t total\t fratio")

	all_signals_list = []
	if(files_to_process > len(file_names)):
		raise(ValueError("Files to process is too large! Given", files_to_process, "Max:", len(file_names)))
	for name in file_names[:files_to_process]:
		base, extension = os.path.splitext(name)

		imageNameFile = im_directory + "/" + name
		maskNameFile = mask_directory + "/mask." + base + ".png"
		gtNameFile = gt_directory + "/gt." + base + ".txt"

		signals = extract_signals_im_training(imageNameFile, maskNameFile, gtNameFile)
		all_signals_list.extend(signals)
	return all_signals_list
	# return maxList,minList,formFactorList,fillingRatioList
	
def create_signal_type_dict(signals_list):
	signal_type_dict = {}
	for signal in signals_list:
		if(signal.signal_type in signal_type_dict):
			signal_type_dict[signal.signal_type]['signal_list'].append(signal)
		else:
			signal_type_dict[signal.signal_type] = {}
			signal_type_dict[signal.signal_type]['signal_list'] = [signal]
	
	for signal_type in signal_type_dict:
		tmp_std = signal_type_dict[signal_type]
		tmp_sl = tmp_std['signal_list']
		tmp_std['max'] = np.max([signal.pixels for signal in tmp_sl])
		tmp_std['min'] = np.min([signal.pixels for signal in tmp_sl])
		tmp_std['fratio'] = np.mean([signal.fratio for signal in tmp_sl])
		tmp_std['ffactor']= np.mean([signal.ffactor for signal in tmp_sl])

	return signal_type_dict

def print_results_signal_type_dict(signal_type_dict):
	print("\t", end='')
	for signal_type in signal_type_dict:
		print("\t",signal_type, end='')
	print()

	for parameter in signal_type_dict[signal_type]:
		if not (parameter=="signal_list"):
			print(parameter+": ", end='')
			if(len(parameter)<5):
				print("\t", end='')
			for signal_type_tmp in signal_type_dict.keys():
				value=signal_type_dict[signal_type_tmp][parameter]
				# print(type(value), end="")
				if (type(value)==np.float64):
					print("\t {0:.1f}".format(value*100),end='%')
				else:
					print("\t",value,end='')
			print()
def main():
	im_directory = "./Dataset/train"
	mask_directory = "./Dataset/train/mask"
	gt_directory = "./Dataset/train/gt"

	print(sys.argv)

	show_dict        = False if(not (len(sys.argv)>1)) else sys.argv[1] in ['True',1]
	files_to_process = -1   if(not (len(sys.argv)>2)) else int(sys.argv[2])	

	signals_list = calculateImagesMetrics(im_directory,mask_directory,gt_directory, files_to_process=files_to_process)
	signal_type_dict = create_signal_type_dict(signals_list)
	if(show_dict): print_results_signal_type_dict(signal_type_dict)

	return signal_type_dict
if __name__ == '__main__':
	main()

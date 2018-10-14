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
	"""
	Basic Signal object that contains main characteristics of a Signal.
	It can optionally contain images of different views of the image
	"""
	def __init__(self):
		#Images
		self.img_orig = None # Original image of the signal (training)
		self.img_mask = None # Mask of the signal (training)
		self.img_pred = None # Predicted signal (Test)

		#Images dir paths
		self.img_orig_path = None
		self.img_mask_path = None
		
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
		"""
		Given the annotations and different images, it builds the object attributes
		"""
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


def extract_signals_im_training(imageNameFile, maskNameFile, gtNameFile):
	"""
	Given the paths of Image (original), Mask (of the signals), and the GT (with the annotations)
	it builds a signal list of the signals in the image
	"""

	image = cv.imread(imageNameFile)
	imageMask = cv.imread(maskNameFile)
	annotations = load_annotations(gtNameFile)

	signal_list = []
	for annotation in annotations:
		signal = Signal()
		signal.build_from_annotation(annotation, image, imageMask)
		signal.img_orig_path = imageNameFile
		signal.img_mask_path = maskNameFile
		signal_list.append(signal)
	return signal_list


def calculateImagesMetrics(im_directory,mask_directory,gt_directory, files_to_process=-1):
	"""
	Returns a list of All different signals object in all the files given the respective directories
	"""
	file_names = sorted(fnmatch.filter(os.listdir(im_directory), '*.jpg'))


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
	"""
	Given a list of signals (built from calculateImagesMetrics), it calculates a dictionary
	that contains max, min, fratio, ffactor of all different types of signals (A, B, C, etc) 
	"""
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
	"""
	Print the dictionary of signals built in create_signal_type_dict
	"""
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
	print("-----------------")

def get_dictionary(show_dict=False):
	from main import CONSOLE_ARGUMENTS
	im_directory = CONSOLE_ARGUMENTS.im_directory
	mask_directory = CONSOLE_ARGUMENTS.mask_directory
	gt_directory = CONSOLE_ARGUMENTS.gt_directory
	files_to_process = CONSOLE_ARGUMENTS.numFiles


	signals_list = calculateImagesMetrics(im_directory,mask_directory,gt_directory, files_to_process=files_to_process)
	signal_type_dict = create_signal_type_dict(signals_list)
	if(show_dict): print_results_signal_type_dict(signal_type_dict)

	return signal_type_dict

def test_metrics():
	from main import CONSOLE_ARGUMENTS
	print(CONSOLE_ARGUMENTS)
	show_dict = CONSOLE_ARGUMENTS.printT1
	files_to_process = CONSOLE_ARGUMENTS.numFiles

	return get_dictionary(show_dict=show_dict, files_to_process=files_to_process)
	
if __name__ == '__main__':
    # read arguments
    from main import parse_arguments
    parse_arguments()
    test_metrics()

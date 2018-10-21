#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage:
  traffic_sign_detection.py <dirName> <outPath> <pixelMethod> [--windowMethod=<wm>] 
  traffic_sign_detection.py -h | --help
Options:
  --windowMethod=<wm>        Window method       [default: 'None']
"""


# from docopt import docopt
import numpy as np
import fnmatch
import os
import imageio
from evaluation.load_annotations import load_annotations
import evaluation.evaluation_funcs as evalf
from metrics import get_dictionary
from split import divide_training_validation_SL

from tqdm import tqdm

from preprocess import preprocess_image
from candidate_generation_pixel import candidate_generation_pixel
from morphology import apply_morphology
from candidate_generation_window import generate_windows
from window_filter import filter_windows
from template_matching import template_matching

import cv2 as cv
import pickle as pckl
import time

import glob 

def msk2rgb(msk):
    msk = msk.astype('uint8')
    # pixel_candidates = remove_small_noise(pixel_candidates)
    msk = np.dstack([msk]*3)
    # immask = msk*im
    # cv.imshow("test.png",immask)
    # cv.imshow("imageb",im)
    # cv.waitKey(0)

    return msk

def get_pixel_candidates(filepath):
    from main import CONSOLE_ARGUMENTS
    
    directory = CONSOLE_ARGUMENTS.im_directory
    output_dir = CONSOLE_ARGUMENTS.out_directory
    
    pixel_selector = CONSOLE_ARGUMENTS.pixel_selector
    preprocess = CONSOLE_ARGUMENTS.prep_pixel_selector
    morphology = CONSOLE_ARGUMENTS.morphology
    boundingBox = CONSOLE_ARGUMENTS.boundingBox
    reduce_bbs = CONSOLE_ARGUMENTS.reduce_bbs
    window_filter = CONSOLE_ARGUMENTS.window_filter
    view_img = CONSOLE_ARGUMENTS.view_imgs
    nar = CONSOLE_ARGUMENTS.non_affinity_removal
    
    print("FILEPATH:",filepath)
    _, name = filepath.rsplit('/', 1)
    base, extension = os.path.splitext(name)
    imageNameFile = directory + "/" + base+extension
    im = imageio.imread(imageNameFile)	
    
    prep_im = preprocess_image(im, preprocess)
    msk = candidate_generation_pixel(prep_im, pixel_selector)
    msk = apply_morphology(msk, morphology)
    bb_list = generate_windows(msk, boundingBox, reduce_bbs=reduce_bbs)
    msk, bb_list = filter_windows(bb_list, msk, window_filter)
    rgb_msk = msk2rgb(msk)
    bb_class_list = template_matching(msk, bb_list, nar)
    
    output_dir_selector = output_dir+"/"+pixel_selector
    args_tuple = (pixel_selector, preprocess, morphology, boundingBox, reduce_bbs, window_filter)
    fd = '{}/'.format(output_dir_selector)
    for arg in args_tuple:
        fd+='{}_'.format(arg)
    fd = fd[:-1]
#    print("----------")
#    print(fd)
    if not os.path.exists(fd):
        os.makedirs(fd)
    im_out_path_name = fd + "/" + "mask."+base+".png"
    pkl_out_path_name = fd + "/" + "mask."+base+".pkl"
    imageio.imwrite(im_out_path_name, np.uint8(np.round(rgb_msk)))
    pkl_bb_list = None
    if(bb_class_list is not None): pkl_bb_list = convertBBFormat(bb_class_list)
    pckl_file = open(pkl_out_path_name,"wb")
    pckl.dump(pkl_bb_list,pckl_file)
    pckl_file.close()

    if(view_img):
        pc_copy = msk.copy()
        immask = np.dstack([pc_copy]*3)*im
        if(pc_copy.max() == 1): pc_copy*=255
        if(bb_list is not None):
            for x,y,w,h,name in bb_class_list:
                cv.rectangle(pc_copy,(x,y),(x+w,y+h),(200,0,0),2)
                cv.rectangle(immask,(x,y),(x+w,y+h),(200,0,0),2)
                cv.putText(pc_copy,name,(x,y), cv.QT_FONT_NORMAL, 1,(150,150,150),2,cv.LINE_AA)
        small_pc = cv.resize(pc_copy, (0,0), fx=0.5, fy=0.5)
        small_im = cv.resize(immask, (0,0), fx=0.5, fy=0.5)
        cv.imshow('window1',small_pc)
        cv.imshow('imres',small_im)
        
        k = cv.waitKey()
        if k==27: # Esc key to stop
            exit()
                
    return msk, bb_class_list

    
def traffic_sign_detection(directory, output_dir, pixel_method, window_method):
    file_names = sorted(fnmatch.filter(os.listdir(directory), '*.jpg'))
    file_names = glob.glob(directory+"/*.jpg")
    print("DIRECTORY:", directory)
    
    for filepath in tqdm(file_names, ascii=True, desc="Generating masks"):
        base, name = filepath.split("\\")
        filepath=base+"/"+name
        rgb_mask, __ = get_pixel_candidates(filepath)

def traffic_sign_detection_test(directory, output_dir, pixel_method, window_method, use_dataset="training"):
    """
	Calculates all statistical evaluation metrics of different pixel selector method (TRAINING AND VALIDATION)
	* Inputs:
	- directory = path to train images
	- outpit_dir = Directory where to store output masks, etc. For instance '~/m1-results/week1/test'
	- pixel_method = pixel method that will segmentate the image
    - window_method = -------
	*Outputs:
	- pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy
	"""
    pixelTP  = 0
    pixelFN  = 0
    pixelFP  = 0
    pixelTN  = 0

    windowTP = 0
    windowFN = 0
    windowFP = 0

    window_precision = 0
    window_accuracy  = 0
    window_sensitivity  = 0

    # print("splitting in trainning test")
    # Load image names in the given directory
    # file_names = sorted(fnmatch.filter(os.listdir(directory), '*.jpg'))
    
    signals_type_dict = get_dictionary()
    
    training, validation = [], []
    for key in signals_type_dict:
        sig_subdict = signals_type_dict[key] 
        training_type, validation_type = divide_training_validation_SL(sig_subdict['signal_list'])
        training.extend(training_type)
        validation.extend(validation_type)

    # print("extracting mask")
    dataset = training
    if(use_dataset == 'validation'):
        dataset = validation
    # if(CONSOLE_ARGUMENTS.use_test):
    totalTime = 0
    dataset_paths = [signal.img_orig_path for signal in dataset]
    
    for signal_path in tqdm(dataset_paths, ascii=True, desc="Calculating Statistics"):
        startTime = time.time()
        rgb_mask, bb_list = get_pixel_candidates(signal_path)
        totalTime = time.time() - startTime
        
        if(bb_list is not None): bb_list = convertBBFormat(bb_list)
        _, name = signal_path.rsplit('/', 1)
        base, extension = os.path.splitext(name)
        # Accumulate pixel performance of the current image #################
        pixel_annotation = imageio.imread('{}/mask/mask.{}.png'.format(directory,base)) > 0

        [localPixelTP, localPixelFP, localPixelFN, localPixelTN] =\
        evalf.performance_accumulation_pixel(rgb_mask, pixel_annotation)
        pixelTP = pixelTP + localPixelTP
        pixelFP = pixelFP + localPixelFP
        pixelFN = pixelFN + localPixelFN
        pixelTN = pixelTN + localPixelTN
        
        [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity] =\
        evalf.performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)

        if(bb_list != None):
            # Accumulate object performance of the current image ################
            window_annotationss = load_annotations('{}/gt/gt.{}.txt'.format(directory, base))                

            [localWindowTP, localWindowFN, localWindowFP] = \
            evalf.performance_accumulation_window(bb_list, window_annotationss)
            windowTP = windowTP + localWindowTP
            windowFN = windowFN + localWindowFN
            windowFP = windowFP + localWindowFP

            # Plot performance evaluation
            [window_precision, window_sensitivity, window_accuracy] = \
            evalf.performance_evaluation_window(windowTP, windowFN, windowFP)
    
    print("meanTime", totalTime/len(dataset))
    print("pixelTP", pixelTP, "\t", pixelFP, "\t", pixelFN)
    print("windowTP", windowTP, "\t", windowFP, "\t", windowFN)
    return [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity,\
            window_precision, window_accuracy, window_sensitivity]


def convertBBFormat(bb_list):
    out_list = list()
    for x,y,w,h,__ in bb_list:
        out_list.append([y,x,y+h,x+w])
    return out_list


if __name__ == '__main__':
    # read arguments
    from main import CONSOLE_ARGUMENTS
    use_dataset = CONSOLE_ARGUMENTS.use_dataset
    images_dir = CONSOLE_ARGUMENTS.im_directory        
    output_dir = CONSOLE_ARGUMENTS.out_directory        
    pixel_method =  CONSOLE_ARGUMENTS.pixel_selector
    print(images_dir,output_dir,pixel_method)
    window_method = 'None'
    print("Performing TSD with",use_dataset,"dataset")
    traffic_sign_detection(images_dir, output_dir, pixel_method, window_method)



    

    

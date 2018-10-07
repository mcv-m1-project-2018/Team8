#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Usage:
  traffic_sign_detection.py <dirName> <outPath> <pixelMethod> [--windowMethod=<wm>] 
  traffic_sign_detection.py -h | --help
Options:
  --windowMethod=<wm>        Window method       [default: 'None']
"""


from docopt import docopt
import numpy as np
import fnmatch
import os
import sys
import imageio
from candidate_generation_pixel import candidate_generation_pixel
from candidate_generation_window import candidate_generation_window
from evaluation.load_annotations import load_annotations
import evaluation.evaluation_funcs as evalf
from metrics import get_dictionary
from split import divide_training_validation_SL

from argparse import ArgumentParser


def traffic_sign_detection(directory, output_dir, pixel_method, window_method):

    pixelTP  = 0
    pixelFN  = 0
    pixelFP  = 0
    pixelTN  = 0

    windowTP = 0
    windowFN = 0
    windowFP = 0

    window_precision = 0
    window_accuracy  = 0

    print("splitting in trainning test")
    # Load image names in the given directory
    file_names = sorted(fnmatch.filter(os.listdir(directory), '*.jpg'))
    
    signals_type_dict = get_dictionary()
    
    training, validation = [], []
    for key in signals_type_dict:
        sig_subdict = signals_type_dict[key]
        training_type, validation_type = divide_training_validation_SL(sig_subdict['signal_list'])
        training.extend(training_type)
        validation.extend(validation_type)

    print("extracting mask")
    for signal in training:
        signal_path = signal.img_orig_path
        _, name = signal_path.rsplit('/', 1)
        base, extension = os.path.splitext(name)

        # Read file
        im = imageio.imread('{}/{}'.format(directory,name))
        print ('{}/{}'.format(directory,name))

        # Candidate Generation (pixel) ######################################
        pixel_candidates = candidate_generation_pixel(im, pixel_method)

        
        fd = '{}/{}_{}'.format(output_dir, pixel_method, window_method)
        if not os.path.exists(fd):
            os.makedirs(fd)
        
        out_mask_name = '{}.png'.format(fd, base)
        imageio.imwrite(out_mask_name, np.uint8(np.round(pixel_candidates)))

        
        if window_method != 'None':
            window_candidates = candidate_generation_window(im, pixel_candidates, window_method) 
        
        # Accumulate pixel performance of the current image #################
        pixel_annotation = imageio.imread('{}/mask/mask.{}.png'.format(directory,base)) > 0

        [localPixelTP, localPixelFP, localPixelFN, localPixelTN] = evalf.performance_accumulation_pixel(pixel_candidates, pixel_annotation)
        pixelTP = pixelTP + localPixelTP
        pixelFP = pixelFP + localPixelFP
        pixelFN = pixelFN + localPixelFN
        pixelTN = pixelTN + localPixelTN
        
        [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity] = evalf.performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN)

        if window_method != 'None':
            # Accumulate object performance of the current image ################
            window_annotationss = load_annotations('{}/gt/gt.{}.txt'.format(directory, base))
            [localWindowTP, localWindowFN, localWindowFP] = evalf.performance_accumulation_window(window_candidates, window_annotationss)
            windowTP = windowTP + localWindowTP
            windowFN = windowFN + localWindowFN
            windowFP = windowFP + localWindowFP


            # Plot performance evaluation
            [window_precision, window_sensitivity, window_accuracy] = evalf.performance_evaluation_window(windowTP, windowFN, windowFP)
    
    return [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy]





                
if __name__ == '__main__':
    # read arguments

    parser = ArgumentParser()
    parser.add_argument("-pT1", "--printT1", dest="printT1", action="store_true",
                        help="Show the metrics required in Task 1")
    parser.add_argument("-pT2", "--printT2", dest="printT2", action="store_true",
                    help="Show the lenghts before and after the split in Training and Validation of Task 2, \
                    also shows the percentages and mean of pixels in each part respectively")

    parser.add_argument("-nf", "--numberFiles", dest="numFiles", type=int,
                        help="Number of files to process in (Task 1)", default=-1)
    
    parser.add_argument("-imdir", "--im_directory", dest="im_directory",type=str,
                        help="Path to training dataset folder", default="./Dataset/train")
    parser.add_argument("-mkdir", "--mask_directory", dest="mask_directory",type=str,
                        help="Path to training mask folder", default="./Dataset/train/mask")
    parser.add_argument("-gtdir", "--gt_directory", dest="gt_directory",type=str,
                        help="Path to groundtruth dataset folder", default="./Dataset/train/gt")
    parser.add_argument("-outdir", "--out_directory", dest="out_directory",type=str,
                        help="Path to output dataset folder", default="./Dataset/train/gt")


    parser.add_argument("-hist", "--histograms", dest="do_histograms", action="store_true",
                        help="Create Histograms of signals", default=False)
    parser.add_argument("-histNorm", "--histogramNorm", dest="histogram_norm", action="store_true",
                        help="Normalize color before doing histograms", default=False)
    
    parser.add_argument("-ps", "--pixel_selector", dest="pixel_selector",type=str,
                        help="Pixel selector function", default="luv-rgb")
    parser.add_argument("-pps", "--prep_pixel_selector", dest="prep_pixel_selector",type=list,
                        help="Preprocesses to do before pixel selector function", default=["blur"])

    args = parser.parse_args()

    images_dir = args.im_directory         # Directory with input images and annotations
                                            # For instance, '../../DataSetDelivered/test'
    output_dir = args.out_directory         # Directory where to store output masks, etc. For instance '~/m1-results/week1/test'

    pixel_method =  args.pixel_selector

    print(images_dir,output_dir,pixel_method)

    window_method = 'None'
    pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy =\
         traffic_sign_detection(images_dir, output_dir, pixel_method, window_method)

    pixel_fmeasure = 2*((pixel_precision)/(pixel_precision+pixel_sensitivity))

    print("Pixel Precision", pixel_precision)
    print("Pixel Accuracy", pixel_accuracy)
    print("Pixel specificity", pixel_specificity)
    print("Pixel sensitivity", pixel_sensitivity)
    print("Pixel F1-Measure", pixel_fmeasure)
    print("Window Precision: ", window_precision, "Window accuracy", window_accuracy)

    

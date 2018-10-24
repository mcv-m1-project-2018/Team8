import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle
from configobj import ConfigObj
import os
from math import floor

HIST_NAMES = ["simple", "subimage", "pyramid"]

def pyramidHistograms(image, binNum, levels, colorSpace="RGB"):
    imageHist = list()
    for i in range(1,levels):
        imageHist.append(subImageHistograms(image, binNum, 2**i, colorSpace))
    return imageHist


def subImageHistograms(image, binNum, subdivision, colorSpace="RGB"):
    w, h, _ = image.shape
    print(w,h)
    imageHist = list()
    for i in range(subdivision):
        for j in range(subdivision):
            x1 = floor(i*(w/subdivision))
            y1 = floor(j*(h/subdivision))
            x2 = floor((i+1)*(w/subdivision))
            y2 = floor((j+1)*(h/subdivision))
            
            subImage = image[x1:x2,y1:y2,:]
            hist = generateHistogram(subImage, binNum, colorSpace)
            
            imageHist.append(hist)
    return imageHist
    

def generateHistogram(image, binNum, colorSpace="RGB"):
    """
    Plots histograms in the color space selected of all the signals pixels values.
    Main use for profiling mask margins
    * Inputs:
    - imPath = path to train images
    - gtPath = path to annotations
    - maskPath = path to masks
    *Outputs:
    - None
    """
    if (colorSpace == "LAB"):
        image = cv.cvtColor(image, cv.COLOR_BGR2Lab)
    if (colorSpace == "Luv"):
        image = cv.cvtColor(image, cv.COLOR_BGR2Luv)
    elif (colorSpace == "HSL"):
        image = cv.cvtColor(image, cv.COLOR_BGR2HLS)
    elif (colorSpace == "HSV"):
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    elif (colorSpace == "Yuv"):
        image = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    elif (colorSpace == "XYZ"):
        image = cv.cvtColor(image, cv.COLOR_BGR2XYZ)
    elif (colorSpace == "YCrCb"):
        image = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)

    imageHist = []
    for i in range(3):
        histr = cv.calcHist(image, [i], None, [binNum], [0, 256])
        imageHist.append(histr)
    return imageHist


def visualizeHistogram(histogram):
    for j, imageHist in enumerate(histogram):
        color = ('b', 'g', 'r')
        plt.figure()
        plt.title("histogram")
        plt.ioff()
        for i, col in enumerate(color):
            plt.subplot(2,2,i+1)
            plt.plot(imageHist[i], color=col)
            plt.xlim([0, 60])
        plt.subplot(2,2,4)
        for i, col in enumerate(color):
            plt.plot(imageHist[i], color=col)
            plt.xlim([0, 60])
        plt.show()

        # directory = dir + "/" + colorSpace
        # if not os.path.exists(directory):
        # os.makedirs(directory)
        # plt.savefig(directory + "/norm_" + titles[j] + ".png")


def processHistogram(file_names, imPath, config):
    """
    Performs every colorspace histogram
    """
    color_space = config['Histograms']['color_space']
    hist_mode = config['Histograms']['histogram_mode']
    subdivision = config.get('Histograms').as_int('subdivision')
    levels = config.get('Histograms').as_int('levels')
    bin_num = config.get('Histograms').as_int('bin_num')

    print(color_space)    
    histAll = list()
    if(hist_mode not in HIST_NAMES):
        raise(ValueError("Hist mode ot recognized", hist_mode, \
                         " VS. ", HIST_NAMES))
    for name in file_names[:-1]:

        imageNameFile = imPath + "/" + name
        image = cv.imread(imageNameFile)

        if(hist_mode == "simple"):
            imageHist = generateHistogram(image, bin_num, color_space)
        elif(hist_mode == "subimage"):
            imageHist = subImageHistograms(image, bin_num, subdivision, color_space)
        elif(hist_mode == "pyramid"):
            imageHist = pyramidHistograms(image, bin_num, levels, color_space)
        histAll.append(imageHist)
    
    if(config.get('Histograms').as_bool('visualize')):
        visualizeHistogram([histAll[0]])
    return histAll

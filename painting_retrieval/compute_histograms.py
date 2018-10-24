import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle
from configobj import ConfigObj
import os
from math import floor

def pyramidHistograms(image, levels, colorSpace="RGB"):
    imageHist = list()
    for i in range(1,levels):
        imageHist.append(subImageHistograms(image,2*i, colorSpace))
    return imageHist


def subImageHistograms(image, subdivision, colorSpace="RGB"):
    w, h, _ = image.shape
    print(w,h)
    imageHist = list()
    for i in range(subdivision):
        for j in range(subdivision):
            subImage = image[floor(i*(w/subdivision)):floor((i+1)*(w/subdivision)),floor(j*(h/subdivision)):floor((j+1)*(h/subdivision)),:]
            imageHist.append(generateHistograms(subImage, colorSpace))
    return imageHist
    

def generateHistograms(image, colorSpace="RGB"):
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
        histr = cv.calcHist(image, [i], None, [60], [0, 256])
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

    print(color_space)    
    histAll = list()
    for name in file_names[:-1]:

        imageNameFile = imPath + "/" + name
        image = cv.imread(imageNameFile)

        if(hist_mode == "simple"):
            imageHist = generateHistograms(image, color_space)
        elif(hist_mode == "subimage"):
            imageHist = subImageHistograms(image, subdivision, color_space)
        elif(hist_mode == "pyramid"):
            imageHist = pyramidHistograms(image, levels, color_space)
        histAll.append(imageHist)
    
    if(config.get('Histograms').as_bool('visualize')):
        visualizeHistogram([histAll[0]])
    return histAll

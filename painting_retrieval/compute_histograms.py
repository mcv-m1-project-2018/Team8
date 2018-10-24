import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle
from configobj import ConfigObj
import os

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

    print(color_space)    
    histAll = list()
    for name in file_names[:-1]:
        base, extension = os.path.splitext(name)

        imageNameFile = imPath + "/" + name
        image = cv.imread(imageNameFile)

        if(hist_mode == "simple"):
            imageHist = generateHistograms(image, color_space)
        elif(hist_mode == "subimag4e"):
            pass
        elif(hist_mode == "pyramid"):
            pass
        histAll.append(imageHist)
    
    if(config.get('Histograms').as_bool('visualize')):
        visualizeHistogram([histAll[0]])
    return histAll

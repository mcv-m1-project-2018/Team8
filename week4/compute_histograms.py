import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle
from configobj import ConfigObj
import os
from math import floor
from tqdm import tqdm

HIST_NAMES = ["simple", "subimage", "pyramid", "pyramidFast"]

def pyramidHistograms(image, binNum, levels, colorSpace="RGB"):
    imageHist = list()
    for i in range(0,levels):
        subHists = subImageHistograms(image, binNum, 2**i, colorSpace=colorSpace)
        imageHist.append(subHists)
    return imageHist

def accBackpropagationHistograms(image, binNum, levels, colorSpace="RGB"):
    def accBack(histsIndexed, level):
        hi = histsIndexed
        subLvlHists = dict()
        for x in range(0, level, 2):
            for y in range(0, level, 2):
                newHist = hi[(x,y)] + hi[(x,y+1)] + hi[(x+1,y)] + hi[(x+1,y+1)]
                subLvlHists[(int(x/2), int(y/2))] = newHist
        return subLvlHists
    
    def getOrderedHist(subHistsDict, lvl):
        orderedHists = list()
        for x in range(lvl):
            for y in range(lvl):
                orderedHists.append(subHistsDict[(x,y)])
        return orderedHists
        
#    w, h, _ = image.shape
    imageHist = list()
    hasFinished = False
    level = levels
    subHists = subImageHistogramsIndexed(image, binNum, 2**level, colorSpace=colorSpace)
    
    while(not hasFinished):
        orderedHistList = getOrderedHist(subHists, 2**level)
        imageHist.append(orderedHistList)
        subHists = accBack(subHists, 2**level)
        level-=1
        hasFinished = (len(subHists) == 1)
    imageHist = [subHists[(0,0)]] + imageHist
    return imageHist[::-1]
        
            
def subImageHistogramsIndexed(image, binNum, subdivision, colorSpace="RGB"):
    w, h, _ = image.shape
    # print(w,h)
    imageHist = dict()
    for i in range(subdivision):
        x1 = floor(i*(w/subdivision))
        x2 = floor((i+1)*(w/subdivision))
        for j in range(subdivision):
            y1 = floor(j*(h/subdivision))
            y2 = floor((j+1)*(h/subdivision))
            
            subImage = image[x1:x2,y1:y2,:]
            hist = generateHistogram(subImage, binNum, colorSpace=colorSpace)
            
            imageHist[(i, j)] = hist
    return imageHist
   
def subImageHistograms(image, binNum, subdivision, colorSpace="RGB"):
    w, h, _ = image.shape
    # print(w,h)
    imageHist = list()
    for i in range(subdivision):
        x1 = floor(i*(w/subdivision))
        x2 = floor((i+1)*(w/subdivision))
        for j in range(subdivision):
            y1 = floor(j*(h/subdivision))
            y2 = floor((j+1)*(h/subdivision))
            
            subImage = image[x1:x2,y1:y2,:]
            hist = generateHistogram(subImage, binNum, colorSpace=colorSpace)
            
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
        histr = [x[0] for x in histr]
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
    hist_mode   = config['Histograms']['histogram_mode']
    subdivision = config.get('Histograms').as_int('subdivision')
    levels      = config.get('Histograms').as_int('levels')
    bin_num     = config.get('Histograms').as_int('bin_num')

    print(color_space)    
    histAll = list()
    if(hist_mode not in HIST_NAMES):
        raise(ValueError("Hist mode ot recognized", hist_mode, \
                         " VS. ", HIST_NAMES))
    for name in tqdm(file_names):

        imageNameFile = imPath + "/" + name
        image = cv.imread(imageNameFile)

        if(hist_mode == "simple"):
            imageHist = generateHistogram(image, bin_num, color_space)
        elif(hist_mode == "subimage"):
            imageHist = subImageHistograms(image, bin_num, subdivision, color_space)
        elif(hist_mode == "pyramid"):
            imageHist = pyramidHistograms(image, bin_num, levels, color_space)
        elif(hist_mode == "pyramidFast"):
            imageHist = accBackpropagationHistograms(image,bin_num, levels, color_space)
        histAll.append(imageHist)
    
    if(config.get('Histograms').as_bool('visualize')):
        visualizeHistogram([histAll[0]])
    return histAll

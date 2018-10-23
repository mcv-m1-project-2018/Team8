import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle
import fnmatch
import os
from configobj import ConfigObj

def visualizeHistograms(imPath, colorSpace="RGB"):
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
    file_names = (fnmatch.filter(os.listdir(imPath), '*.jpg'))

    histAll = list()
    for name in file_names[:-1]:
        base, extension = os.path.splitext(name)

        imageNameFile = imPath + "/" + name
        image = cv.imread(imageNameFile)

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

        color = ('b', 'g', 'r')
        imageHist = []
        for i in range(3):
            histr = cv.calcHist(image, [i], None, [60], [0, 256])
            imageHist.append(histr)
        histAll.append(imageHist)
    print(histAll)
    # for j, imageHist in enumerate(histAll):
        # color = ('b', 'g', 'r')
        # plt.figure()
        # plt.title(file_names[j])
        # plt.ioff()
        # for i, col in enumerate(color):
            # plt.plot(imageHist[i], color=col)
            # plt.xlim([0, 60])
            # plt.show()
        # directory = dir + "/" + colorSpace
        # if not os.path.exists(directory):
        #    os.makedirs(directory)
        # plt.savefig(directory + "/norm_" + titles[j] + ".png")


def do_hists():
    """
    Performs every colorspace histogram
    """
    config = ConfigObj('./Test.config')
    museum_set_random = config['Directories']['imdir_in']
    color_space = config['color_space']
    # if color_space == None:
        # color_space = ["RGB", "LAB", "Luv", "HSL", "HSV", "Yuv", "XYZ", "YCrCb"]

    # for color in color_space:
    print(color_space)
    visualizeHistograms(museum_set_random, color_space)


# if __name__ == '__main__':
    # do_hists()


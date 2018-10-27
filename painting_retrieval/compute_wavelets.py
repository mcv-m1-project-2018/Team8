import pywt 
import numpy as np
import cv2 as cv
from tqdm import tqdm
from matplotlib import pyplot as plt

def computeWaveletTransform(image, levels, method):
    coeffs = pywt.wavedec2(image, method, level=levels)
    return coeffs


def processWavelets(imPath, filenames, levels, method):
    waveAll = list()
    for name in tqdm(filenames):
        imageNameFile = imPath + "/" + name
        image = cv.imread(imageNameFile)

        waveAll.append(computeWaveletTransform(image, levels, method))
    return waveAll



# image = cv.imread("./Dataset/museum_set_random/ima_000000.jpg")
# histogram = calculateGranulometry(image,50)
# cv.imshow("asd",image)
# visualizeHistogram(histogram,50)
# cv.waitKey()
import pywt 
import numpy as np
import cv2 as cv
from tqdm import tqdm
from matplotlib import pyplot as plt
from compute_granulometry import calculateGranulometry

def computeWaveletTransform(image, levels, method):
    coeffs = pywt.wavedec2(image, method, level=levels)
    return coeffs


def processWavelets(filenames, imPath, levels, method, n_bin):
    waveAll = []
    for name in tqdm(filenames):
        waveImg = []
        imageNameFile = imPath + "/" + name
        image = cv.imread(imageNameFile)

        coeff = computeWaveletTransform(image, levels, method)

        caN = coeff[0]
        del coeff[0]
        waveImg.append(calculateGranulometry(caN,n_bin))

        for cHn, cVn, cDn in coeff:
            waveImg.append(calculateGranulometry(cHn,n_bin))
            waveImg.append(calculateGranulometry(cVn,n_bin))
            waveImg.append(calculateGranulometry(cDn,n_bin))
    return waveAll



# image = cv.imread("./Dataset/museum_set_random/ima_000000.jpg")
# histogram = calculateGranulometry(image,50)
# cv.imshow("asd",image)
# visualizeHistogram(histogram,50)
# cv.waitKey()
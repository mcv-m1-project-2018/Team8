import pywt 
import numpy as np
import cv2 as cv
from tqdm import tqdm

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



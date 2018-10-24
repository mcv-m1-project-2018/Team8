import numpy as np
import cv2 as cv
import ml_metrics as metrics
import numpy as np

def euclidean_distance(t_bins, q_bins):
    for j in range(len(t_bins)):
        euclidean_dist = sqrt((t_bins(1) - q_bins(1)) ^ 2 + (t_bins(2) - q_bins(2)) ^ 2 +
                              (t_bins(3) - q_bins(3)) ^ 2)
        return euclidean_dist

def l_1_distance():


def x_sq():


def hist_intersect():


def kernhell():
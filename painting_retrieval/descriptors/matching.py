# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:53:54 2018

@author: hamdd
"""
import cv2 as cv
import statistics
from tqdm import tqdm
import numpy as np

def imageSimilarityAdd(match_list):
    return sum(x.distance for x in match_list)

def imageSimilarityAddK(match_list, k):
    sortedList = sorted(match_list, key=lambda x: x.distance)
    return imageSimilarityAdd(sortedList[:k])

def imageSimilarityMean(match_list):
    return statistics.mean(x.distance for x in match_list)

def imageSimilarityMeanK(match_list, k):
    sortedList = sorted(match_list, key=lambda x: x.distance)
    statistics.mean(x.distance for x in sortedList[:k])


matching_s = {
    "add": imageSimilarityAdd,
    "mean": imageSimilarityMean
}

def matching_query(all_desc_t, all_desc_q, matching, distance_method, k = -1):
    """
    Return a list of matches of one image for all query images
    
    matching_type can be: 
    BruteForce
    BruteForce-L1
    BruteForce-Hamming
    BruteForce-Hamming(2)
    FlannBased

    """

    matcher = cv.DescriptorMatcher()
    matcher = matcher.create(matching)
    all_match = list()
    all_dist = list()
    all_sortIndex = list()

    for desc_q in tqdm(all_desc_q,desc="Matching"):
        queryMatchList = list()
        queryDistanceList = list()
        for desc_t in all_desc_t:
            matches = matcher.match(desc_t,desc_q)
            queryMatchList.append(matches)
            if(k != -1):
                matches = sorted(matches, key=lambda x: x.distance)[:k]
            matches = matching_s[distance_method](matches)

            queryDistanceList.append(matches)

        index_highest = np.argsort(queryDistanceList)[:k]
        all_sortIndex.append(index_highest)
        all_match.append(queryMatchList)
        all_dist.append(queryDistanceList)

    return all_match, all_dist, all_sortIndex
        
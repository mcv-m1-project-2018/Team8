# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:53:54 2018

@author: hamdd
"""
import cv2 as cv
import statistics
from tqdm import tqdm
import numpy as np

def imageSimilarityAdd(dist_list):
    return sum(dist_list)

# def imageSimilarityAddK(match_list, k):
#     sortedList = sorted(match_list, key=lambda x: x.distance)
#     return imageSimilarityAdd(sortedList[:k])

def imageSimilarityMean(dist_list):
    return statistics.mean(dist_list)

# def imageSimilarityMeanK(match_list, k):
#     sortedList = sorted(match_list, key=lambda x: x.distance)
#     statistics.mean(x.distance for x in sortedList[:k])

def imageSimilarityLen(dist_list):
    return len(dist_list)


matching_s = {
    "add": imageSimilarityAdd,
    "mean": imageSimilarityMean,
    "Len": imageSimilarityLen
}

def matching_query(all_desc_t, all_desc_q, matching, distance_method, k = -1, th=-1):
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
            # mean([x.distance for x in match_list])
            if(th > 0):
                dist_list = [x.distance for x in matches if x.distance < th]
            else:
                dist_list = [x.distance for x in matches]
            distance = matching_s[distance_method](dist_list)

            queryDistanceList.append(distance)

        index_highest = np.argsort(queryDistanceList)[:k]
        all_sortIndex.append(index_highest)
        all_match.append(queryMatchList)
        all_dist.append(queryDistanceList)

    return all_match, all_dist, all_sortIndex
        
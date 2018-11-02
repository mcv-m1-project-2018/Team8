# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:53:54 2018

@author: all
"""
import cv2 as cv
import statistics
from tqdm import tqdm, trange
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
    "len": imageSimilarityLen
}

def matching_query(all_desc_t, all_desc_q, matching, distance_method,k=5, k_distance = -1, th_distance=-1, th_discarted=500):
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

    for desc_q in tqdm(all_desc_q,desc="Matching Total"):
        queryMatchList = list()
        queryDistanceList = list()
        for desc_t in tqdm(all_desc_t,desc="Matching of one query"):
            matches = matcher.match(desc_q,desc_t)
            queryMatchList.append(matches)
            if(k_distance != -1):
                matches = sorted(matches, key=lambda x: x.distance)[:k_distance]
            # mean([x.distance for x in match_list])
            if(th_distance > 0):
                dist_list = [x.distance for x in matches if x.distance < th_distance]
            else:
                dist_list = [x.distance for x in matches]

            distance = matching_s[distance_method](dist_list)

            queryDistanceList.append(distance)
        if(distance_method=="len"):
            index_highest = np.argsort(queryDistanceList)[::-1][:k]
        else:
            index_highest = np.argsort(queryDistanceList)[:k]
            
        if(distance_method == "mean" and th_discarted > 0):
            if( min(queryDistanceList) > th_discarted):
                index_highest = [-1]
        if(distance_method == "len" and th_discarted > 0):
            if( max(queryDistanceList) < th_discarted):
                index_highest = [-1]
        all_sortIndex.append(index_highest)
        all_match.append(queryMatchList)
        all_dist.append(queryDistanceList)

    return all_match, all_dist, all_sortIndex
        
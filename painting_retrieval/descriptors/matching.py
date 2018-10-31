# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 19:53:54 2018

@author: hamdd
"""
import cv2 as cv

def imageSimilarityAdd(match_list):
    value = sum(match_list, key=lambda x: x.distance)
    np.sum()

def imageSimilarityAddK(match_list, k):
    newlist = sorted(match_list, key=lambda x: x.distance)

def imageSimilarityMean(match_list):
    newlist = sorted(match_list, key=lambda x: x.distance)



def matching_query(all_desc_t, all_desc_q, matching):
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
    for desc_q in all_desc_q:
        queryList = list()
        for desc_t in all_desc_t:
            matches = matcher.match(desc_t,desc_q)
            queryList.append(matches)
            imageSimilarity(matches)
        all_match.append(queryList)

    return all_match
        
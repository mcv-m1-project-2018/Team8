
from evaluation_histograms import evaluateHistograms, evaluatesubImageHistograms, evaluatePyramidHistograms
import numpy as np 
from tqdm import tqdm

def evaluateQuery(t_img_list, q_img, eval_method, hist_meth):
    distance_list = []
    for t_image in t_img_list:
        dist = 0
        if(hist_meth == "simple"):
            dist = evaluateHistograms(t_image, q_img, eval_method)
        elif(hist_meth == "subimage"):
            dist = evaluatesubImageHistograms(t_image, q_img, eval_method)
        elif(hist_meth == "pyramid"):
            dist = evaluatePyramidHistograms(t_image, q_img, eval_method)
        distance_list.append(dist[0])
    return distance_list

def evaluateQueryTest(t_img_list, q_img_list, k, eval_method, hist_method):
    similarity = []
    distAllList = list()
    for query_img in tqdm(q_img_list):
        dist_list = evaluateQuery(t_img_list, query_img, eval_method, hist_method)
        
        #Get k-images' index with highest similarity
        index_highest = np.argsort(dist_list)[:k]
        similarity.append(index_highest)
        distAllList.append(dist_list)
    return distAllList, similarity
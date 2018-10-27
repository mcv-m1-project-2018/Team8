from ml_metrics.average_precision import mapk
import cv2 as cv
import numpy as np
correlation_images = {0: 76, 1: 105, 2: 34, 3: 83, 4: 109, 5: 101, 6: 57, 7: 27, 8: 50, 9: 84, 10: 25, 11: 60, 12: 45, \
    13: 99, 14: 107, 15: 44, 16: 65, 17: 63, 18: 111, 19: 92, 20: 67, 21: 22, 22: 87, \
    23: 85, 24: 13, 25: 39, 26: 103, 27: 6, 28: 62, 29: 41}

def evaluate_prediction(query_path, file_query_names, train_path, file_train_names, index_similarity, k):
    query_list = []
    predicted_list = []

    for x in range(len(file_query_names)):
        subpredicted_list = []
        query_list.append(['ima_{:06d}.jpg'.format(correlation_images[x])])
        
        for y in range(min(k,len(index_similarity))): # the min is required to prevent errors if the index similarity is shorter than k
            subpredicted_list.append(file_train_names[index_similarity[x][y]])
        predicted_list.append(subpredicted_list)

    
    return mapk(query_list, predicted_list, k)
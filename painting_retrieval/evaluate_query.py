from ml_metrics.average_precision import mapk
import cv2 as cv
import numpy as np
correlation_images = {0: 30, 1: 102, 2: 100, 3: 94, 4: 56, 5: 10, 6: 101, 7: 0,\
                      8: 107, 9: 82, 10: 108, 11: 106, 12: 12, 13: 78, 14: 63,\
                      15: 97, 16: 34, 17: 47, 18: 21, 19: 15, 20: 68, 21: 46,\
                      22: 26, 23: 32, 24: 75, 25: 19, 26: 57,
            27: 98, 28: 93, 29: 20}
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
from ml_metrics.average_precision import mapk
import cv2 as cv
import numpy as np

correlation_images =   {0: [-1], 1: [-1], 2: [115, 122], 3: [-1], 4: [-1], 5: [99], 6: [-1], 7: [89],\
                        8: [19], 9: [85], 10: [90], 11: [121, 117], 12: [-1], 13: [-1], 14: [130], 15: [6, 84],\
                        16: [35, 48, 52], 17: [118], 18: [-1], 19: [-1], 20: [-1], 21: [-1], 22: [60],\
                        23: [119, 128], 24: [-1], 25: [47], 26: [-1], 27: [41], 28: [-1], 29: [126, 123]}

def evaluate_prediction(query_path, file_query_names, train_path, file_train_names, index_similarity, k):
    query_list = []
    predicted_list = []

    for x in file_query_names:
        x = int(x[4:-4])
        subpredicted_list = []
        query_list.append(correlation_images[x])

        for y in range(min(k,len(index_similarity[x]))): # the min is required to prevent errors if the index similarity is shorter than k
            subpredicted_list.append(index_similarity[x][y])
        predicted_list.append(subpredicted_list)

    
    return mapk(query_list, predicted_list, k)
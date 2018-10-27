from configobj import ConfigObj
from compute_histograms import processHistogram
from compare_images import evaluateQueryTest
from evaluate_query import evaluate_prediction
import fnmatch
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def main():
    config = ConfigObj('./Test.config')

    if(config.as_int('numTrain') != -1):
        numTrain = config.as_int('numTrain')
    else:
        numTrain = None
    if(config.as_int('numQueries') != -1):
        numQueries = config.as_int('numQueries')
    else:
        numQueries = None
    
    train_path = config['Directories']['imdir_train']
    file_train_names = (fnmatch.filter(os.listdir(train_path), '*.jpg'))[:numTrain]

    query_path = config['Directories']['imdir_query']
    file_query_names = (fnmatch.filter(os.listdir(query_path), '*.jpg'))[:numQueries]
    
    histogram_mode = config['Histograms']['histogram_mode'] 

    color_list = ["rgb", "LAB", "Luv", "HSL", "HSV", "Yuv", "XYZ", "YCrCb"]

    for color_space in color_list:
        config['Histograms']['color_space'] = color_space
        histograms_train = processHistogram(file_train_names,train_path, config)
        histograms_query = processHistogram(file_query_names,query_path, config)


        k = config.get('Evaluate').as_int('k')
        eval_method = config['Evaluate']['eval_method']


        distAllList, index_similarity = evaluateQueryTest(histograms_train, histograms_query, k, eval_method, histogram_mode)

        if(config.get('Visualization').as_bool("enabled")):
            # fig, axs = plt.subplots(1,2)
            for x in range(len(file_query_names)):
                q_img = cv.imread(query_path+file_query_names[x])
                # axs[0].imshow(q_img)
                cv.imshow("query",q_img)
                for y in range(k):
                    t_img = cv.imread(train_path+file_train_names[index_similarity[x][y]])
                    # axs[1].imshow(t_img)
                    cv.imshow("result",t_img)
                    key = cv.waitKey()
                    if(key == 27): #ESC for exit
                        break

        print(evaluate_prediction(query_path, file_query_names, train_path, file_train_names, index_similarity, k))



if __name__ == '__main__':
    main()

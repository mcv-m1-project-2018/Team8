from configobj import ConfigObj
from compute_histograms import processHistogram
from preprocess_histograms import preprocessAllHistograms
from compare_images import evaluateQueryTest
from evaluate_query import evaluate_prediction
from compute_wavelets import processWavelets
from compute_granulometry import processGranulometry
import fnmatch
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pickle as pckl

def visualizeQueryResults(query_path,file_query_names, train_path, file_train_names, index_similarity):
    for x in range(len(file_query_names)):
        q_img = cv.imread(query_path+file_query_names[x])
        cv.imshow("query",q_img)
        for y in range(k):
            t_img = cv.imread(train_path+file_train_names[index_similarity[x][y]])
            cv.imshow("result",t_img)
            key = cv.waitKey()
            if(key == 27): #ESC for exit
                break

def getNamesBySimilarity(file_train_names, index_similarity ):
    similarityNames = []
    for image_list in index_similarity:
        similarityByQuery = []
        for i in image_list:
            similarityByQuery.append(file_train_names[i])
        similarityNames.append(similarityByQuery)
    return similarityNames



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

    # for color_space in color_list:
    # config['Histograms']['color_space'] = color_space

    if(config['mode']== "histogram"):
        histograms_train = processHistogram(file_train_names, train_path, config)
        histograms_query = processHistogram(file_query_names, query_path, config)
        preproc_mode = config['Histograms']['preprocess']
        if(preproc_mode  is not "None"):
            histograms_train = preprocessAllHistograms(histograms_train,preproc_mode)
            histograms_train = preprocessAllHistograms(histograms_train,preproc_mode)

    elif(config['mode']== "wavelet"):
        level = config.get('Wavelets').as_int('levels')
        method = config['Wavelets']['method']

        histograms_train = processWavelets(train_path, file_train_names, level, method)
        histograms_query = processWavelets(query_path, file_query_names, level, method)

    elif(config['mode']== "granulometry"):
        bin_num = config.get('Granulometry').as_int('bin_num')
        visualize = config.get('Granulometry').as_bool('visualize')

        histograms_train = processGranulometry(file_train_names, train_path, bin_num, visualize)
        histograms_query = processGranulometry(file_query_names, query_path, bin_num, visualize)

    k = config.get('Evaluate').as_int('k')
    eval_method = config['Evaluate']['eval_method']

    distAllList, index_similarity = evaluateQueryTest(histograms_train, histograms_query, k, eval_method, histogram_mode)



    if(config.get('Visualization').as_bool("enabled")):
        visualizeQueryResults(query_path,file_query_names, train_path, file_train_names, index_similarity)


    print("The Results for the query are",evaluate_prediction(query_path, file_query_names, train_path, file_train_names, index_similarity, k))

    if(config.get('Save_Pickle').as_bool('save')):
        pout = config['Save_Pickle']['pathOut']
        mode = config['mode']
        hmode= config['Histograms']['histogram_mode']
        cs = config['Histograms']['color_space']
        bins = config['Histograms']['bin_num']
        if(hmode in ["pyramid","pyramidFast"]):
            levels = config['Histograms']['levels']
            hmode = str(levels) + hmode
        save_path = pout + "_" + mode + "_"+ hmode + "_"+ cs + "_"+ str(bins) +"bins" +".pkl"
        print("FILENAME:", save_path)
        nameList = getNamesBySimilarity(file_train_names,index_similarity)

        pckl_file = open(save_path,"wb")
        pckl.dump(nameList,pckl_file)
        pckl_file.close()



if __name__ == '__main__':
    main()

from configobj import ConfigObj

from compute_histograms import processHistogram
from preprocess_histograms import preprocessAllHistograms
from compare_images import evaluateQueryTest
from evaluate_query import evaluate_prediction
from compute_wavelets import processWavelets
from compute_granulometry import processGranulometry
from preprocess.detect_textbox import detect_text_meanShift,detect_text_hats,saveTextBoxArray
from preprocess.crop_and_rotate import compute_angles, saveCroppingArray
from preprocess.evaluate_bbox import main_evaluate_bb
    
from descriptors.detect import detect_all_kp
from descriptors.compute import compute_all_features
from descriptors.matching import matching_query

import fnmatch
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pickle as pckl

def visualizeQueryResults(query_path,file_query_names, train_path, file_train_names, index_similarity, k):
    for x in range(len(file_query_names)):
        q_img = cv.imread(query_path+file_query_names[x])
        q_img = cv.resize(q_img,(500,500))
        cv.imshow("query",q_img)
        for y in range(len(index_similarity[x])):
            t_img = cv.imread(train_path+file_train_names[index_similarity[x][y]])
            t_img = cv.resize(t_img,(500,500))
            cv.imshow("result",t_img)
            key = cv.waitKey()
            if(key == 27): #ESC for exit
                break

def getNamesBySimilarity(file_train_names, index_similarity ):
    similarityNames = []
    for image_list in index_similarity:
        similarityByQuery = []
        for i in image_list:
            if(i > 0):
                similarityByQuery.append(file_train_names[i])
            else:
                similarityByQuery.append(None)
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

    if(config['mode'] == "histogram"):
        histograms_train = processHistogram(file_train_names, train_path, config)
        histograms_query = processHistogram(file_query_names, query_path, config)
        preproc_mode = config['Histograms']['preprocess']
        if(preproc_mode  is not "None"):
            histograms_train = preprocessAllHistograms(histograms_train,preproc_mode)
            histograms_train = preprocessAllHistograms(histograms_train,preproc_mode)

    elif(config['mode'] == "wavelet"):
        bin_num = config.get("Granulometry").as_int("bin_num")
        level = config.get('Wavelets').as_int('levels')
        method = config['Wavelets']['method']

        histograms_train = processWavelets(file_train_names, train_path, level, method)
        histograms_query = processWavelets(file_query_names, query_path, level, method)

    elif(config['mode'] == "granulometry"):
        bin_num = config.get('Granulometry').as_int('bin_num')
        visualize = config.get('Granulometry').as_bool('visualize')

        histograms_train = processGranulometry(file_train_names, train_path, bin_num, visualize)
        histograms_query = processGranulometry(file_query_names, query_path, bin_num, visualize)
    elif(config["mode"] == "features"):
        bb_list_t = []
        cropping_list = []
        if(config['Features'].get('preprocess').as_bool("preprocess")):
            cmp_angles = config['Features'].get('preprocess').as_bool("compute_angles")
            d_text_hats= config['Features'].get('preprocess').as_bool("detect_text_hats")
            debug_text_hats = config['Features'].get('preprocess').as_bool("debug_text_hats")
            if(cmp_angles): 
                crop_mth = config['Features']['preprocess']["cropping_method"]
                save_cropping = config['Features']['preprocess']["cropping_list_savepath"]
                crop_debug = config['Features'].get('preprocess').as_bool("compute_angles_debug")
                cropping_list = compute_angles(file_query_names, query_path, crop_mth, crop_debug)
                saveCroppingArray(save_cropping,cropping_list)
            if(d_text_hats):
                save_bb = config['Features']['preprocess']["textBox_save"]
                bb_list_t = detect_text_hats(file_train_names, train_path, debug_text_hats)
#                bb_list_t, bb_mask = detect_text_hats(file_train_names, train_path, debug_text_hats)
                saveTextBoxArray(save_bb,bb_list_t)
#                maskList = bb_mask
            if(config['Features'].get('preprocess').as_bool("evaluate_bb")):
                namefile_pkl = config['Features']['preprocess']["pickle_eval_textBox"]
                main_evaluate_bb(namefile_pkl,bb_list_t)

        img_width = config["Features"].as_int("image_width")
        crop = config['Features'].get('preprocess').as_bool("crop")
        detector = config["Features"]["detect"]
        kp_t, f_t = detect_all_kp(file_train_names, train_path, detector, image_width=img_width, mask=bb_list_t)
        kp_q, f_q = detect_all_kp(file_query_names, query_path, detector, image_width=img_width, rot_rectangle=cropping_list, crop=crop)
        
        computer = config["Features"]["compute"]
        kp_t, desc_t = compute_all_features(file_train_names, train_path, kp_t,\
                                            computer, detector, image_width=img_width)
        kp_q, desc_q = compute_all_features(file_query_names, query_path, kp_q,\
                                            computer, detector,image_width=img_width, \
                                            rot_rectangle=cropping_list, crop=crop)

    
    k = config.get('Evaluate').as_int('k')

    if(config['mode'] in ["granulometry", "histogram", "wavelet"]):
        eval_method = config['Histograms']['eval_method']
        distAllList, index_similarity = evaluateQueryTest(histograms_train, histograms_query, k, eval_method, histogram_mode)
    else:
        matching = config["Features"]["matching"]
        k_match = config.get('Features').as_int('k')
        th_matching = config.get('Features').as_float('th')
        distance_method = config["Features"]["distance"]
        th_disc = config.get('Features').as_float('th_discarted')
        print(detector,computer,matching,k_match,th_matching,distance_method,th_disc)

        all_match, all_dist, index_similarity = matching_query(desc_t, desc_q, matching, distance_method, k, k_distance=k_match, th_distance=th_matching,th_discarted=th_disc)


    if(config.get('Visualization').as_bool("enabled")):
        visualizeQueryResults(query_path,file_query_names, train_path, file_train_names, index_similarity,k)


    print("The Results for the query are",evaluate_prediction(query_path, file_query_names, train_path, file_train_names, index_similarity, k))

    if(config.get('Save_Pickle').as_bool('save')):
        pout = config['Save_Pickle']['pathOut']
        mode = config['mode']
        if(mode != "features"):
            hmode= config['Histograms']['histogram_mode']
            cs = config['Histograms']['color_space']
            bins = config['Histograms']['bin_num']
            evalm = config['Histograms']['eval_method']
            if(hmode in ["pyramid","pyramidFast"]):
                levels = config['Histograms']['levels']
                hmode = str(levels) + hmode
            save_path = pout + evalm +"_" + mode + "_"+ hmode + "_"+ cs + "_"+ str(bins) +"bins" +".pkl"
        else:
            detect = config["Features"]["detect"]
            compute = config["Features"]["compute"]
            matching = config["Features"]["matching"]
            distance = config["Features"]["distance"]
            k_distance = k_match = config['Features']['k']
            th_distance = config['Features']['th']
            th_discarted = config['Features']['th_discarted']
            save_path = pout + detect + "_" + compute + "_" + matching + "_" + distance + "_" + k_distance + "_" + th_distance + "_" + th_discarted + ".pkl"
        print("FILENAME:", save_path)
        nameList = getNamesBySimilarity(file_train_names,index_similarity)
        print(nameList)
        pckl_file = open(save_path,"wb")
        pckl.dump(nameList,pckl_file)
        pckl_file.close()



if __name__ == '__main__':
    main()

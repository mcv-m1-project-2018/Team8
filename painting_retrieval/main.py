from configobj import ConfigObj
from compute_histograms import processHistogram
import fnmatch
import os

def main():
    config = ConfigObj('./Test.config')

    train_path = config['Directories']['imdir_train']
    file_train_names = (fnmatch.filter(os.listdir(train_path), '*.jpg'))

    query_path = config['Directories']['imdir_query']
    file_query_names = (fnmatch.filter(os.listdir(query_path), '*.jpg'))
    
    histograms_train= processHistogram(file_train_names,train_path, config)
    histograms_query = processHistogram(file_query_names,query_path, config)



if __name__ == '__main__':
    main()

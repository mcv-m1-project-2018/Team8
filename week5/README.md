# Project 2: Image retrieval system

[![alt text](http://www.cvc.uab.es/wp-content/uploads/2016/07/copy-logo3.svg "Centre de Visió per Computador")](http://cvc.cat/)

The goal of this project is to learn the basic concepts and techniques to build a simple query by example retrieval system for finding paintings in a museum image collection.

## Install
This project requires opencv, pywt, matplotlib, numpy, **tqdm**, cv2, pickle, fnmatch, configobj, math and ml_metrics.

To install the project just clone the repository and use python (3.6 or 3.7) to run:
```bash
python main.py
```
## Options
### Datasets
Default values of Training and Dataset paths are `./Dataset/museum_set_random`, `./Dataset/query_devel_random` and `./Dataset/query_test_random`. You can change dirs in the Test.config file.

### Test configuration
The Test.config file stores all the arguments for running the program. If the names/values are changed in this file, the code will be run with that configuration.

## Files
### Compute histograms
The file compute_histograms.py generates pyramidal histograms, per channel histograms and block-based histograms (subimage).

### Preprocess histograms
The file preprocess_histograms.py processes all the histograms for evaluating them properly.

### Evaluate histograms
After processing the histograms, an evaluation is needed. The file evaluation_histograms.py file creates a method for comparing two histograms according to the type of histogram (pyramid, subimage or full histograms).

### Evaluation measures
In the evaluation_measures.py file, different image measures have been computed: euclidean, L1, x_sq, hist_int, kernhell, bhattacharyya, x_sq_alt, kl_div and correlate.

### Compare images
The compare_images.py file compares the histograms of two images according to the histogram mode. The query_devel_random and museum_set_random images are compared. This allows us to choose the best evaluation method to apply for the query_test_random images folder.

### Evaluate Query
The evaluate_query.py file includes a dictionary with the image correspondance between the query_devel_random and the museum_set_random images. With this, the precision of the images comparison and selection is computed.

### Extract descriptors
The extract_descriptors.py file consists of a keypoints detector and features extractor method contained in a switcher. It is not used since the utils.py method does the same function. 

### Detect
In the detect.py file, all the keypoints of an image are detected and stored in a list called kp_list.

### Utils
The utils.py file contains two switchers: the detector_s switcher, which contains a library for each detector, and the compute_s switcher, for the feature descriptors.

### Matching
The matching.py file contains three methods for evaluating the similarity between two images. All these methods are based in the distance between the keypoints (mean, sumatory and length).

### Evaluate query
The evaluate_query.py file returns a list of the k most similar images of every image in the database. It uses the mapk function from the ml_metrics library created by Ben Hamner and Wendy Kan.

### Compute
In the compute.py file, the features using the keypoints and descriptors are computed. and stored in a list.

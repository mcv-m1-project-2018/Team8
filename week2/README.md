# Project 1: Traffic Sign Detection

[![alt text](http://www.cvc.uab.es/wp-content/uploads/2016/07/copy-logo3.svg "Centre de Visió per Computador")](http://cvc.cat/)

Project 1 aims to create a program capable of detect traffic signs in the road at real-time using classical computer vision tecniques.

## Install
This project requires opencv, matplotlib, numpy, **tqdm** and skimage.

To install the project just clone the repository and use python (3.6 or 3.7) to run:
```bash
python main.py [options]
```
With the options you can activate or deactivate different modules and/or change variables. Main features are explained in Options section.
## Options
### Datasets
Default values of Training and Dataset paths are `./Dataset/train` and `./Dataset/test`. You can change dirs with `-imdir TRAINING_PATH` or `-testdir TEST_PATH` flags respectively.

### Traffic Sign Detection
Using `-ttsd` you can activate the module that directly extracts the pixel candidates. It will automatically show the results of candidates in the Training dataset.

#### Used dataset
You can change the dataset with `-ud` or `--use_dataset` flag, adding `training`, `validation` or `test` to select different datasets. If you select the TEST dataset this will automatically save the results in `./Dataset/test`. You can change the output directory with `-outdir OUTPUT_PATH` flag.

#### Select Candidates ####
You can control diferent predefined thresholds and how the pixel candidates are selected.
##### Color Mask #####
With `-ps [rgb|luv|hsv|hsv-rgb|lab|luv-rgb|GW-luv-rgb|luv-hsv|normRGB-luv-rgb]` you can change the pixel selector (mask creation) that is specially handpicked for each colorspace. Note that some pixel selector has preprocess of the image in it. 

##### Image Preprocess #####
You can add more preprocess steps using the flag `-pps [PREPROCESS [PREPROCESS [...]]`. Where `PREPROCESS` can be `[neutralize|grayWorld|whitePatch|normrgb|blur]`. 

##### Morphology #####
Different sets of morphology operations can be controled with `-m [MORPHOLOGY [MORPHOLOGY [...]]`, where `MORPHOLOGY` can be `[m1]`.

##### Window generation #####
Once the image is binarized, small and odd objects can be deleted creating a window for each individual object and comparing their caracteristics with a standard defined in a study on the training metrics. To activate this function use `-w [CCL_WIN_TYPE [CCL_WIN_TYPE [...]]` where `CCL_WIN_TYPE` can be `[m1]`.

##### Sliding Window #####
Once the image is binarized, small and odd objects can be deleted using Sliding Window processing. To activate this function use `-sw [S_WIN_TYPE [S_WIN_TYPE [...]]` where `S_WIN_TYPE` can be `[m1]`.

#### Other variables
`-nf NUMBER` Number of files to process from `-imdir`.
`-mkdir MASK_DIR_PATH` GT masks.
`-gtdir ANNOTATIONS_PATH` path where you extract the annotation text files.
`-outdir OUTPUT_PATH` output results path.
`-wf WINDOW` Window filtering method.
`-swsize SLIDING_WINDOW_SIZE` Size of the sliding window.
`-rbbs REDUCE_BB` Reduce Bounding Boxes size.
`-vi VIEW_IMAGES` View images?
`-nar NON_AFFINITY_REMOVAL` Delete objects that doesn't look like any signal?
`-nar_1 NON_AFFINITY_REMOVAL_ARG1` Threshold to delete low affinity in small signals.
`-nar_2 NON_AFFINITY_REMOVAL_ARG2` Threshold to delete low affinity in big signals. 

## Test Modules 
#### Metrics
You can see an analysis of annotation features (task 1) activating the `-tm` flag.

#### Split
You can se an analysis of the split in Training and Validation datasets using `-ts`.

#### Traffic Sign Detection
You can se an analysis of the split in Training and Validation datasets using `-ttsd`.

#### Histograms
You can analyze the histogram of every type of signal (A,B,C,D,E,F) in different colorspaces (using the `-ps` flag) and different color preprocess (using `-pps` flag) in the Histogram Module. To activate this module you have to use `-hist` flag. 

## Examples
Examples of code can be:

### Example 1
Saving results of `luv-rgb` with `blur` and `grayWorld` preprocesses using the `test` dataset. 

```bash
python main.py -ttsd -ud test -ps luv-rgb -pps blur grayWorld
```

## Example 2
Example 1 with only 100 files and using morphology method 1, and window generator method 1.
```bash
python main.py -ttsd -nf 100 -ps luv-rgb -pps blur -m m1 -w m1
```

## Example 3
Example 2 with ... add more examples for our beloved readers!

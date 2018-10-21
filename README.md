# Team 8 Module 1 repository [OUTDATED]

## Introduction
This project aims to select pixel candidates taking into account only the color of certain pixels without going further (no neighbourhood color analysis). You can select many different colorspaces and correct the color in different ways in order to improve statistical results (precission, accuracy, specificity, sensitivity, F1-measure).

To run the project use:
```bash
python main.py [options]
```
With the options you can activate or deactivate different modules and/or change variables. Main features are explained in Options section.
# Options
## Datasets
Default values of Training and Dataset paths are `./Dataset/train` and `./Dataset/test`. You can change dirs with `-imdir TRAINING_PATH` or `-testdir TEST_PATH` flags respectively.

## Traffic Sign Detection
Using `-ttsd` you can activate the module that directly extracts the pixel candidates. It will automatically show the results of candidates in the Training dataset.

#### Used dataset
The dataset can be changed with using `-ud` or `--use_dataset` flag, adding `training`, `validation` or `test` to select different datasets. If you select the TEST dataset this will automatically save the results in `./Dataset/test`. You can change the output directory with `-outdir OUTPUT_PATH` flag.

#### Select Candidates ####
You can control diferent predefined thresholds and how the pixel candidates are selected.

##### Color Mask #####
With `-ps [rgb|luv|hsv|hsv-rgb|lab|luv-rgb|GW-luv-rgb|luv-hsv|normRGB-luv-rgb]` the pixel selector (mask creation) can be changed, which is specially handpicked for each colorspace. Note that some pixel selector has preprocess of the image in it. 

##### Image Preprocess #####
More preprocess steps can be added using the flag `-pps [PREPROCESS [PREPROCESS [...]]`. Where `PREPROCESS` can be `[neutralize|grayWorld|whitePatch|normrgb|blur]`. 

##### Morphology #####
Different sets of morphology operations can be controled with `-m [MORPHOLOGY [MORPHOLOGY [...]]`, where `MORPHOLOGY` can be `[m1]`.

##### CCL Window Processing #####
Once the image is binarized, small and odd objects can be deleted using CCL Window processing (or Sliding Window seen in next paragraph). To activate this function use `-w [CCL_WIN_TYPE [CCL_WIN_TYPE [...]]` where `CCL_WIN_TYPE` can be `[m1]`.

##### Sliding Window #####
Once the image is binarized, small and odd objects can be deleted using Sliding Window processing. To activate this function use `-sw [S_WIN_TYPE [S_WIN_TYPE [...]]` where `S_WIN_TYPE` can be `[m1]`.

#### Other variables
`-nf NUMBER` Number of files to process from `-imdir`.
`-mkdir MASK_DIR_PATH` GT masks.
`-gtdir ANNOTATIONS_PATH` path where you extract the annotation text files.
`-outdir OUTPUT_PATH` output results path.

## Test Modules 
#### Metrics
An analysis of annotation features (task 1) can be seen by activating the `-tm` flag.

#### Split
An analysis of the split in Training and Validation datasets can be seen by using `-ts`.

#### Traffic Sign Detection
An analysis of the split in Training and Validation datasets can be seen by using `-ttsd`.

#### Histograms
The histogram of every type of signal (A,B,C,D,E,F) can be analyzed in different colorspaces (using the `-ps` flag) and different color preprocess (using `-pps` flag) in the Histogram Module. To activate this module you have to use `-hist` flag. 

# Examples
Examples of code can be:

## Example 1
Saving results of `luv-rgb` with `blur` and `grayWorld` preprocesses using the `test` dataset. 

```bash
python main.py -ttsd -ud test -ps luv-rgb -pps blur grayWorld
```

## Example 2
Add more examples here :)



# Team 8 Module 1 repository

## Introduction
This project aims to select pixel candidates taking into account only the color of certain pixels without going further (no neighbourhood color analysis). You can select many different colorspaces and correct the color in different ways in order to improve statistical results (precission, accuracy, specificity, sensitivity, F1-measure).

To run the projectuse:
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

#### Pixel Selector and Image Preprocess
With `-ps [rgb|luv|lab|luv-rgb|Blur-luv-rgb|normRGB-luv-rgb|GW-Blur-luv-rgb|GW-RGB|WP-RGB]` you can change the pixel selector (mask creation) that is specially handpicked for each colorspace. Note that some pixel selector has preprocess of the image in it. You can add more preprocess steps using the flag `-pps [PREPROCESS[PREPROCESS[...]]`. Where `PREPROCESS` can be `[neutralize|grayWorld|whitePatch|normrgb|blur]`.

#### Other variables
`-nf NUMBER` Number of files to process from `-imdir`.
`-mkdir MASK_DIR_PATH` GT masks.
`-gtdir ANNOTATIONS_PATH` path where you extract the annotation text files.
`-outdir OUTPUT_PATH` output results path.

### Test Modules 
#### Metrics
You can see an analysis of annotation features (task 1) activating the `-tm` flag.

#### Split
You can se an analysis of the split in Training and Validation datasets using `-ts`.

#### Traffic Sign Detection
You can se an analysis of the split in Training and Validation datasets using `-ttsd`.

#### Histograms
You can analyze the histogram of every type of signal (A,B,C,D,E,F) in different colorspaces (using the `-ps` flag) and different color preprocess (using `-pps` flag) in the Histogram Module. To activate this module you have to use `-hist` flag. 




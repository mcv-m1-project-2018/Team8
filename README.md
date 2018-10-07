# Team 8 Module 1 repository

## Introduction
This project aims to select pixel candidates taking into account only the color of certain pixels without going further (no neighbourhood color analysis). You can select many different colorspaces and correct the color in different ways in order to improve statistical results (precission, accuracy, specificity, sensitivity, F1-measure).

To run the projectuse:
```bash
python main.py [options]
```
With the options you can activate or deactivate different modules and/or change variables. Main features are explained in Options section.
## Options
### Traffic Sign Detection
Using `-ttsd` you can activate the module that directly extracts the pixel candidates. It will automatically show the results of candidates in the Training dataset.

You can change the dataset with `-ud` or `--use_dataset` flag, adding `training`, `validation` or `test` to select different datasets. If you select the TEST dataset this will automatically save the results in `./Dataset/test`. You can change the output directory with `-outdir OUTPUT_PATH` flag.


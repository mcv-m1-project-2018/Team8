from metrics import test_metrics
from traffic_sign_detection import traffic_sign_detection_test,traffic_sign_detection
from split import test_split
from histogramVisualization import do_hists

from argparse import ArgumentParser
 
def parse_arguments():
    """
	Parse line arguments
	"""
    parser = ArgumentParser()
    general_args = parser.add_argument_group("General arguments")
    
    module_args_group = parser.add_argument_group("Module arguments")
    module_group = module_args_group.add_mutually_exclusive_group(required=True)

    tsd_args = parser.add_argument_group("Traffic Sign Detection Module arguments")
    hist_args = parser.add_argument_group("Histogram Module arguments")
    # print_group = parser.add_argument_group("Print arguments")
    # print_group.add_argument("-pT1", "--printT1", dest="printT1", action="store_false",
    #                     help="Do not show the metrics required in Task 1")
    # print_group.add_argument("-pT2", "--printT2", dest="printT2", action="store_true",
    #                 help="Do not show the lenghts before and after the split in Training and Validation of Task 2, \
    #                 also shows the percentages and mean of pixels in each part respectively")
    
    module_group.add_argument("-tm", "--test_metrics", dest="tm", action="store_true",
                        help="Activate Module Metrics and get the measures of annotations")
    module_group.add_argument("-ts", "--test_split", dest="ts", action="store_true",
                        help="Activate Module Split and get statistics of training-validation split")
    module_group.add_argument("-ttsd", "--test_traffic_sign_detection", dest="ttsd", action="store_true",
                        help="Activate Module Traffic Sign Detection and get statistical results")
    module_group.add_argument("-hist", "--histograms", dest="hist", action="store_true",
                        help="Activate Module Histograms of Signals and save histogram plots in --", default=False)

    
    hist_args.add_argument("-histNorm", "--histogramNorm", dest="histogram_norm", action="store_true",
                        help="Normalize color before doing histograms", default=False)
    hist_args.add_argument("-outhistdir", "--out_hist_directory", dest="hist_save_directories",type=str,
                        help="Path to output for histogram plots folder", default="./Dataset/histogramNormPrecise/")
    hist_args.add_argument("-csh", "--color_spaces_histograms", dest="csh",nargs='+', choices=["RGB","LAB","Luv","normRGB","HSL","HSV","Yuv","XYZ", "YCrCb"],
                        help="Colorspaces in which signals' histogram will be calculated", type=str, default=None)
     
    tsd_args.add_argument("-ud", "--use_dataset", dest="use_dataset", default="training", choices=['training','validation','test'],
                        help="Which dataset use for trafic sign detection instead of training")
    tsd_args.add_argument("-ps", "--pixel_selector", dest="pixel_selector",type=str,
                        help="Pixel selector function", default="luv-rgb")
    tsd_args.add_argument("-pps", "--prep_pixel_selector", dest="prep_pixel_selector",nargs='+',
                        help="Preprocesses to do before pixel selector function", type=str, default=None)
    tsd_args.add_argument("-m", "--morphology", dest="morphology",nargs='+',
                        help="Morphology method", type=str, default=None)
    tsd_args.add_argument("-bb", "--bounding_box", dest="boundingBox",nargs='+',
                        help="Bounding Box extractor type", type=str, default=None)
    tsd_args.add_argument("-w", "--window", dest="window",nargs='+',
                        help="Window CCL method", type=str, default=None)
    

    general_args.add_argument("-nf", "--numberFiles", dest="numFiles", type=int,
                        help="Number of files to process in (Task 1)", default=-1)
    general_args.add_argument("-imdir", "--im_directory", dest="im_directory",type=str,
                        help="Path to training dataset folder", default="./Dataset/train")
    general_args.add_argument("-mkdir", "--mask_directory", dest="mask_directory",type=str,
                        help="Path to training mask folder", default="./Dataset/train/mask")
    general_args.add_argument("-gtdir", "--gt_directory", dest="gt_directory",type=str,
                        help="Path to groundtruth dataset folder", default="./Dataset/train/gt")
    general_args.add_argument("-outdir", "--out_directory", dest="out_directory",type=str,
                        help="Path to output dataset folder", default="./Dataset/output/maskOut/")
    general_args.add_argument("-testdir", "--test_directory", dest="test_directory",type=str,
                        help="Path of input test dataset folder", default="./Dataset/test")

    return parser.parse_args()


CONSOLE_ARGUMENTS = parse_arguments()


def main():
    """
    Main script to redirect modules
    """
    global CONSOLE_ARGUMENTS

    if CONSOLE_ARGUMENTS.tm:
        test_metrics()
    if CONSOLE_ARGUMENTS.ts:
        test_split()
    if CONSOLE_ARGUMENTS.ttsd:
        use_dataset = CONSOLE_ARGUMENTS.use_dataset
        images_dir = CONSOLE_ARGUMENTS.im_directory         # Directory with input images and annotations
        output_dir = CONSOLE_ARGUMENTS.out_directory        # Directory where to store output masks, etc. For instance '~/m1-results/week1/test'
        pixel_method =  CONSOLE_ARGUMENTS.pixel_selector

        print(images_dir,output_dir,pixel_method)
        window_method = 'None'
        if(use_dataset in ["training","validation"]):
            #Carregar variables
            #Executar funció
            pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity, window_precision, window_accuracy =\
            traffic_sign_detection_test(images_dir, output_dir, pixel_method, window_method, use_dataset=use_dataset)
            
            #Printar resultats
            pixel_fmeasure = 2*((pixel_precision*pixel_sensitivity)/(pixel_precision+pixel_sensitivity))
            print("Pixel Precision:\t", "{0:.1f}".format(pixel_precision*100),end='%\n')
            print("Pixel Accuracy: \t", "{0:.1f}".format(pixel_accuracy*100),end='%\n')
            print("Pixel specificity:\t","{0:.1f}".format(pixel_specificity*100),end='%\n')
            print("Pixel sensitivity:\t","{0:.1f}".format(pixel_sensitivity*100),end='%\n')
            print("Pixel F1-Measure:\t","{0:.1f}".format(pixel_fmeasure*100),end='%\n')
            print("Window Precision:\t","{0:.1f}".format(window_precision*100),end='%\n')
            print("Window accuracy:\t","{0:.1f}".format(window_accuracy*100),end='%\n' )
        else:
            print("Performing TSD with",use_dataset,"dataset")
            traffic_sign_detection(images_dir, output_dir, pixel_method, window_method)
    if CONSOLE_ARGUMENTS.hist:
        do_hists()

if __name__ == '__main__':

    main()
		
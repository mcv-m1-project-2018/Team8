from metrics import test_metrics
from traffic_sign_detection import test_tsd

from argparse import ArgumentParser
 
def parse_arguments():
    parser = ArgumentParser()
    test_group = parser.add_argument_group("Test modules arguments")
    print_group = parser.add_argument_group("Print arguments")
    main_group = parser.add_argument_group("Optional main arguments")
    color_test_group = parser.add_argument_group("Optional color testing arguments")

    print_group.add_argument("-pT1", "--printT1", dest="printT1", action="store_false",
                        help="Do not show the metrics required in Task 1")
    print_group.add_argument("-pT2", "--printT2", dest="printT2", action="store_true",
                    help="Do not show the lenghts before and after the split in Training and Validation of Task 2, \
                    also shows the percentages and mean of pixels in each part respectively")
    
    test_group.add_argument("-tm", "--test_metrics", dest="tm", action="store_true",
                        help="Test metric measures")
    test_group.add_argument("-ttsd", "--test_traffic_sign_detection", dest="ttsd", action="store_true",
                        help="Test traffic sign detection measures")
    test_group.add_argument("-uv", "--use_validation", dest="use_validation", action="store_true",
                        help="Use validation dataset for trafic sign detection instead of training")
                    
    main_group.add_argument("-nf", "--numberFiles", dest="numFiles", type=int,
                        help="Number of files to process in (Task 1)", default=-1)
    main_group.add_argument("-imdir", "--im_directory", dest="im_directory",type=str,
                        help="Path to training dataset folder", default="./Dataset/train")
    main_group.add_argument("-mkdir", "--mask_directory", dest="mask_directory",type=str,
                        help="Path to training mask folder", default="./Dataset/train/mask")
    main_group.add_argument("-gtdir", "--gt_directory", dest="gt_directory",type=str,
                        help="Path to groundtruth dataset folder", default="./Dataset/train/gt")
    main_group.add_argument("-outdir", "--out_directory", dest="out_directory",type=str,
                        help="Path to output dataset folder", default="./Dataset/train/maskOut")
    
    main_group.add_argument("-ps", "--pixel_selector", dest="pixel_selector",type=str,
                        help="Pixel selector function", default="luv-rgb")
    main_group.add_argument("-pps", "--prep_pixel_selector", dest="prep_pixel_selector",nargs='+',
                        help="Preprocesses to do before pixel selector function", type=str)


    color_test_group.add_argument("-hist", "--histograms", dest="do_histograms", action="store_true",
                        help="Create Histograms of signals", default=False)
    color_test_group.add_argument("-histNorm", "--histogramNorm", dest="histogram_norm", action="store_true",
                        help="Normalize color before doing histograms", default=False)
    return parser.parse_args()


CONSOLE_ARGUMENTS = parse_arguments()


def main():
    global CONSOLE_ARGUMENTS

    if CONSOLE_ARGUMENTS.tm:
        test_metrics()

    if CONSOLE_ARGUMENTS.ttsd:
        test_tsd()

if __name__ == '__main__':

    main()
		
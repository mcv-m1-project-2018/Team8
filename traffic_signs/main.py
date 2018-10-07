from argparse import ArgumentParser




def main():
    parser = ArgumentParser()
    parser.add_argument("-pT1", "--printT1", dest="printT1", action="store_true",
                        help="Show the metrics required in Task 1")
    parser.add_argument("-pT2", "--printT2", dest="printT2", action="store_true",
                    help="Show the lenghts before and after the split in Training and Validation of Task 2, \
                    also shows the percentages and mean of pixels in each part respectively")

    parser.add_argument("-nf", "--numberFiles", dest="numFiles", action="store_const",
                        help="Number of files to process in (Task 1)")
    
    parser.add_argument("-imdir", "--im_directory", dest="im_directory", action="store_const",
                        help="Path to training dataset folder")
    parser.add_argument("-mkdir", "--mask_directory", dest="mask_directory", action="store_const",
                        help="Path to training mask folder")
    parser.add_argument("-gtdir", "--gt_directory", dest="gt_directory", action="store_const",
                        help="Path to groundtruth dataset folder")
    
    parser.add_argument("-hist", "--histograms", dest="do_histograms", action="store_true",
                        help="Create Histograms of signals")
    parser.add_argument("-histNorm", "--histogramNorm", dest="histogram_norm", action="store_true",
                        help="Normalize color before doing histograms")
    
    

    parser.add_argument("-q", "--quiet",
                        action="store_false", dest="verbose", default=True,
                        help="don't print status messages to stdout")

if __name__ == '__main__':
	main()
		
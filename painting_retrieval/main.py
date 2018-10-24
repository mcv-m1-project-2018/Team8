from configobj import ConfigObj
from compute_histograms import processHistogram

def main():
    config = ConfigObj('./Test.config')
    histograms = processHistogram(config)

if __name__ == '__main__':
    main()

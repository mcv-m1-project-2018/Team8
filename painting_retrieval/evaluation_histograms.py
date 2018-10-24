from  evaluation_measures import evaluate


def evaluatePyramidHistograms(t_histList, q_histList, method_name):
    dist = 0
    for t_level, q_level in zip(t_histList,q_histList):
        dist += evaluatesubImageHistograms(t_level, q_level, method_name)
    return dist

def evaluatesubImageHistograms(t_histList, q_histList, method_name):
    dist = 0
    for t_subImage, q_subimage in zip(t_histList,q_histList):
        dist += evaluateHistograms(t_subImage, q_subimage, method_name)
    return dist

def evaluateHistograms(t_histList, q_histList, method_name):
    dist = 0
    for ch in range(len(t_histList)):
        dist += evaluate(t_histList[ch], q_histList[ch], method_name)
    return dist



from preprocess.evaluation_funcs import performance_accumulation_window
import pickle as pkl

original_bb_list_gt = [(95, 867, 767, 991), (130, 588, 984, 740), (70, 489, 395, 552), (128, 1072, 649, 1188), (149, 868, 1114, 1044), (160, 763, 753, 879), (92, 629, 272, 673), (149, 55, 584, 140), (199, 978, 825, 1094), (227, 183, 1062, 328), (131, 116, 1076, 281), (121, 0, 876, 145), (124, 37, 493, 122), (151, 133, 1096, 298), (91, 480, 450, 565), (192, 172, 895, 317), (50, 59, 375, 122), (134, 140, 529, 225), (151, 5, 891, 87), (159, 9, 1134, 110), (165, 3, 435, 47), (276, 811, 947, 896), (223, 862, 1059, 978), (287, 71, 970, 156), (160, 26, 795, 107), (121, 972, 804, 1057), (233, 960, 870, 1045), (117, 575, 414, 631), (140, 996, 823, 1081), (139, 709, 498, 765), (183, 50, 1121, 166), (247, 692, 930, 777), (276, 89, 947, 174), (137, 695, 1051, 811), (108, 1, 729, 86), (174, 725, 502, 781), (179, 521, 476, 577), (170, 988, 729, 1073), (119, 860, 421, 916), (193, 151, 762, 236), (130, 583, 515, 639), (288, 225, 711, 281), (174, 42, 746, 110), (152, 127, 1082, 223), (183, 1134, 850, 1217), (106, 136, 917, 221), (177, 81, 928, 162), (162, 563, 504, 619), (87, 578, 482, 634), (210, 116, 595, 172), (212, 105, 607, 161), (202, 35, 937, 120), (138, 49, 556, 105), (114, 13, 850, 99), (114, 771, 913, 855), (128, 1099, 797, 1184), (173, 1053, 745, 1121), (155, 733, 467, 777), (133, 534, 934, 619), (157, 77, 1023, 159), (196, 138, 1062, 220), (122, 848, 694, 916), (57, 497, 475, 553), (142, 452, 494, 508), (78, 634, 501, 690), (261, 80, 930, 165), (108, 469, 420, 513), (121, 504, 506, 560), (202, 80, 1068, 162), (225, 817, 643, 873), (135, 3, 553, 59), (270, 84, 688, 140), (166, 782, 1096, 878), (156, 756, 745, 819), (94, 534, 743, 619), (142, 139, 893, 220), (138, 79, 531, 132), (234, 704, 903, 789), (152, 649, 903, 730), (213, 59, 936, 144), (223, 656, 956, 741), (257, 476, 1243, 592), (117, 100, 401, 144), (196, 655, 919, 740), (166, 53, 545, 109), (100, 489, 412, 545), (192, 635, 781, 720), (96, 30, 460, 86), (209, 516, 1195, 632), (276, 31, 1262, 147), (277, 68, 970, 153), (154, 206, 823, 291), (54, 71, 413, 124), (158, 160, 891, 243), (171, 17, 772, 102), (128, 913, 661, 976), (131, 495, 664, 558), (180, 155, 1202, 271), (206, 600, 1192, 716), (208, 85, 542, 141), (154, 191, 619, 254), (147, 849, 397, 893), (178, 864, 1165, 988), (92, 24, 421, 80), (68, 33, 366, 89), (150, 671, 1045, 772), (45, 36, 379, 92), (133, 314, 383, 358), (87, 163, 710, 248), (137, 877, 988, 986), (208, 25, 1195, 170), (114, 874, 1009, 975), (153, 615, 730, 700), (255, 232, 1335, 408), (356, 2229, 2740, 2525), (281, 2162, 2143, 2398), (278, 486, 2433, 751), (255, 12, 2210, 337), (345, 2148, 2468, 2413), (344, 53, 2339, 318), (408, 2181, 3623, 2566), (362, 10, 2829, 302), (411, 2334, 2854, 2625), (354, 260, 2453, 513), (319, 2022, 2794, 2347), (296, 2382, 2567, 2654), (380, 2960, 2367, 3225), (317, 170, 2160, 435), (308, 2440, 2295, 2705), (373, 300, 2816, 591), (350, 65, 2863, 385), (389, 39, 2468, 275), (368, 374, 2551, 579), (398, 2133, 2773, 2329), (231, 565, 2018, 770), (324, 394, 2297, 599), (350, 102, 2646, 338), (264, 500, 2119, 658), (252, 362, 2225, 567), (334, 500, 1929, 639), (302, 245, 2329, 450), (367, 218, 2726, 454), (372, 425, 2129, 630), (292, 2436, 2265, 2641), (419, 40, 3324, 365), (327, 2034, 2406, 2270), (412, 1737, 3075, 2033), (251, 2161, 2254, 2354), (437, 116, 3332, 350), (450, 1881, 2809, 2117), (408, 15, 2591, 220), (412, 413, 2511, 678), (291, 2091, 2390, 2356), (393, 2174, 2768, 2499), (349, 340, 2246, 576), (281, 2272, 2416, 2487), (269, 2422, 1900, 2615), (271, 3219, 2406, 3434), (390, 2169, 3233, 2554), (176, 21, 1501, 197), (441, 25, 3050, 381), (343, 1945, 2700, 2241), (437, 58, 2812, 383), (294, 60, 2625, 325), (484, 1892, 3179, 2217), (340, 317, 2097, 522), (345, 2471, 2486, 2767), (391, 2431, 2288, 2667), (254, 2693, 2011, 2898), (182, 2545, 991, 2661), (379, 1796, 2486, 2061), (242, 2277, 1807, 2442), (368, 310, 2698, 546), (368, 176, 3004, 440), (341, 38, 2320, 243), (324, 2590, 2183, 2795), (352, 1340, 2515, 1576), (396, 1909, 2762, 2205), (405, 2636, 2519, 2872), (427, 1471, 3135, 1767), (317, 534, 2638, 830), (361, 1810, 2986, 2135), (244, 255, 1833, 460), (336, 2056, 2315, 2261), (269, 1975, 1216, 2091), (296, 3304, 2167, 3497), (348, 2846, 2165, 3051), (311, 4, 2607, 240), (470, 82, 3089, 347), (394, 2981, 2367, 3186), (445, 41, 3273, 277), (261, 309, 2294, 514), (369, 484, 2284, 660), (336, 1591, 2632, 1827), (370, 37, 2661, 242), (387, 61, 2420, 266), (296, 507, 2527, 712), (254, 205, 1734, 325), (401, 572, 2374, 777), (407, 2335, 2773, 2571), (396, 2015, 2451, 2191), (394, 433, 2367, 638), (250, 2499, 2019, 2638), (434, 1555, 2467, 1760), (324, 2105, 2790, 2313)]

def load_bboxes(filename):
    pckl_file = open(filename,"wb+")
    bb_list = pkl.load(pckl_file)
    pckl_file.close()
    return bb_list

def convertBBFormat(bb_list):
    out_list = list()
    for x,y,w,h in bb_list:
        # out_list.append([y,x,y+h,x+w])
        out_list.append([x,y,x+w,y+h])
    return out_list

def evaluate_bb(bb_list_gt, bb_list_annotation):
    windowTP = 0
    windowFN = 0
    windowFP = 0
    for i,bb_annot in enumerate(bb_list_annotation):
        bb_gt = bb_list_gt[i]
        [localWindowTP, localWindowFN, localWindowFP] = performance_accumulation_window([bb_gt], [bb_annot])

        windowTP = windowTP + localWindowTP
        windowFN = windowFN + localWindowFN
        windowFP = windowFP + localWindowFP

    print("windowTP",windowTP,"windowFN",windowFN,"windowFP",windowFP)

def main_evaluate_bb(filenamePkl, bb_list_annotation):
    # bb_list_gt = load_bboxes(filenamePkl)
    bb_list_gt = original_bb_list_gt
    new_bb_list_annot = convertBBFormat(bb_list_annotation)
    evaluate_bb(bb_list_gt,new_bb_list_annot)
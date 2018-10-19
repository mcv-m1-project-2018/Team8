#!/usr/bin/python

def reduce_win_size(w, img):
    small_img = img[w[1]:(w[1]+w[3]),w[0]:(w[0]+w[2])]
    column = small_img.sum(0)
    row =    small_img.sum(1)
    
    new_x = 0
    new_w = 1
    new_y = 0
    new_h = 1

    while not column[ new_x]: new_x+=1
    while not column[-new_w]: new_w+=1
    while not    row[ new_y]: new_y+=1
    while not    row[-new_h]: new_h+=1

    new_win = (w[0] + new_x, w[1] + new_y, w[2] - new_w - new_x + 1, w[3] - new_h - new_y+1)
    
    return new_win

def reduce_winds_sizes(windows, img):
    for i, window in enumerate(windows):
        windows[i] = reduce_win_size(window, img)
    return windows
    
    
def candidate_generation_window_example1(im, pixel_candidates):
    window_candidates = [[17.0, 12.0, 49.0, 44.0], [60.0, 90.0, 100.0, 130.0]]

    return window_candidates
 
def candidate_generation_window_example2(im, pixel_candidates):
    window_candidates = [[21.0, 14.0, 54.0, 47.0], [63.0, 92.0, 103.0, 132.0], [200.0, 200.0, 250.0, 250.0]]

    return window_candidates
 
# Create your own candidate_generation_window_xxx functions for other methods
# Add them to the switcher dictionary in the switch_method() function
# These functions should take an image, a pixel_candidates mask (and perhaps other parameters) as input and output the window_candidates list.

def switch_method(im, pixel_candidates, method):
    switcher = {
        'example1': candidate_generation_window_example1,
        'example2': candidate_generation_window_example2
    }
    # Get the function from switcher dictionary
    func = switcher.get(method, lambda: "Invalid method")

    # Execute the function
    window_candidates = func(im, pixel_candidates)

    return window_candidates

def candidate_generation_window(im, pixel_candidates, method):

    window_candidates = switch_method(im, pixel_candidates, method)

    return window_candidates

    
if __name__ == '__main__':
    window_candidates1 = candidate_generation_window(im, pixel_candidates, 'example1')
    window_candidates2 = candidate_generation_window(im, pixel_candidates, 'example2')

    

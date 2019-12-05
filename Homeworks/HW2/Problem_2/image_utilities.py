import cv2
import numpy as np
import time
from Problem_2.graph_cut_processing import *

# Source: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
from Problem_2.histogram import compute_pdf


def image_resize(image, width=None, height=700, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    if(h < 700):
        return image

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

#Source: https://stackoverflow.com/questions/41925853/fill-shapes-contours-using-numpy
def fill_contours(arr):
    return np.maximum.accumulate(arr,1) & \
           np.maximum.accumulate(arr[:,::-1],1)[:,::-1]

# Source: https://stackoverflow.com/questions/45020672/convert-pyqt5-qpixmap-to-numpy-ndarray
def segment_image(modified_image,input_image_filename):
    modified_image.save('./tmp/modified_image.png')
    time.sleep(.300)
    modified_image = cv2.imread('./tmp/modified_image.png').astype(np.float32)
    input_image = cv2.imread(input_image_filename)
    input_image = image_resize(input_image).astype(np.float32)
    cv2.imwrite("./tmp/input_image_resized.png",input_image) # for testing only
    mask = np.abs(modified_image - input_image)
    cv2.imwrite('./tmp/mask.png',mask)
    return segment_image_utility(input_image,modified_image)

def segment_image_utility(input_image,modified_image):
    m,n,_ = input_image.shape
    diff_img = np.abs(modified_image - input_image)

    diff_img[np.where((diff_img <= [1, 1, 1]).all(axis=2))] = [0, 0, 0]
    modified_image[np.where((diff_img == [0, 0, 0]).all(axis=2))] = [0, 0, 0]
    b, g, r = cv2.split(modified_image)

    # Fill in B to fill in garbage mask
    b = (b // 255).astype(np.bool)
    b = fill_contours(b)

    # Negate the mask
    b = np.logical_not(b)
    b = (b * 255).astype(np.uint8)

    # Write to disk (for diagnosing)
    cv2.imwrite("./tmp/image_analysis_output_b.png", b)
    cv2.imwrite("./tmp/image_analysis_output_g.png", g)
    cv2.imwrite("./tmp/image_analysis_output_r.png", r)

    # Create foreground and background masks
    foreground = g
    background = b + r
    cv2.imwrite("./tmp/image_analysis_background.png", background)
    F_prop, B_prop = compute_pdf(input_image,foreground,background)
    nodeids, g = build_graph(input_image, foreground, background, F_prop, B_prop)
    segmented_matte = cut_graph(nodeids, g, input_image)
    cv2.imwrite("./tmp/image_analysis_segmented_image_matte.png", segmented_matte.astype(np.uint8) * 255)
    segmented_image = (input_image * segmented_matte.reshape(m,n,1)).astype(np.uint8)
    cv2.imwrite("./tmp/image_analysis_segmented_image.png", segmented_image)
    return segmented_image

import cv2
import numpy as np
from Problem_2.graph_cut_processing import build_graph, cut_graph
from Problem_2.histogram import compute_pdf


#Source: https://stackoverflow.com/questions/41925853/fill-shapes-contours-using-numpy
def fill_contours(arr):
    return np.maximum.accumulate(arr,1) & \
           np.maximum.accumulate(arr[:,::-1],1)[:,::-1]

def main():
    im1 = cv2.imread("./tmp/input_image_resized.png").astype(np.float32)
    im2 = cv2.imread("./tmp/modified_image.png").astype(np.float32)
    im3 = np.abs(im2 - im1)

    im3[np.where((im3 <= [1, 1, 1]).all(axis=2))] = [0, 0, 0]
    im2[np.where((im3 == [0,0,0]).all(axis=2))] = [0, 0, 0]
    b,g,r = cv2.split(im2)

    # Fill in B to fill in garbage mask
    b = (b//255).astype(np.bool)
    b = fill_contours(b)

    # Negate the mask
    b = np.logical_not(b)
    b = (b*255).astype(np.uint8)

    # Write to disk (for diagnosing)
    cv2.imwrite("./tmp/image_analysis_output_b.png", b)
    cv2.imwrite("./tmp/image_analysis_output_g.png", g)
    cv2.imwrite("./tmp/image_analysis_output_r.png", r)

    # Create foreground and background masks
    foreground = g
    background = b + r
    cv2.imwrite("./tmp/image_analysis_background.png",background)
    compute_pdf(im1,foreground,background)
    # nodeids, g = build_bayes_graph(im1, foreground, background)
    # segmented_im = cut_graph(nodeids, g, im1)
    # cv2.imwrite("./tmp/image_analysis_segmented_image.png",segmented_im)

if __name__=="__main__":
    main()
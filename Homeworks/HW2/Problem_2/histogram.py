import numpy as np
import cv2

def build_histogram(image, mask):
    scribbles = image[mask == 255].reshape((-1, 3))
    num_pixels = len(scribbles)
    H, edges = np.histogramdd(scribbles, bins=(256, 256, 256))
    H = H/num_pixels
    return H

def build_prob_matrix(image,hist):
    M, N, _ = image.shape
    prob = np.zeros((M, N))
    for row in range(0,M):
        for column in range(0,N):
            b,g,r = image[row,column]
            prob[row,column] = hist[int(b%256),int(g%256),int(r%256)]
    return prob

def compute_pdf(image,foreground,background):
    F_hist = build_histogram(image,foreground) * 255
    B_hist = build_histogram(image,background) * 255
    F_prop = build_prob_matrix(image,F_hist)
    B_prop = build_prob_matrix(image,B_hist)
    return F_prop, B_prop

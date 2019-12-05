from numpy import *
from sklearn.naive_bayes import GaussianNB
import maxflow
import cv2
from sklearn.mixture import GaussianMixture as GM
import numpy as np

""" 
Graph Cut image segmentation using max-flow/min-cut. 
"""

import numpy as np
from collections import defaultdict

# Source: https://sandipanweb.wordpress.com/2018/02/11/interactive-image-segmentation-with-graph-cut/
def build_graph(im, foreground_, background_, Fprob, Bprop, sigma=10, lam=1):
    """    Build a graph from 4-neighborhood of pixels.
        Foreground and background is determined from
        labels (1 for foreground, -1 for background, 0 otherwise)
        and is modeled with naive Bayes classifiers."""

    m, n = im.shape[:2]
    #
    # # RGB for foreground and background training data
    # background_data = im[background_ == 255].reshape((-1, 3))
    # foreground_data = im[foreground_ == 255].reshape((-1, 3))
    # background_labels = np.array([0 for i in range(background_data.shape[0])]).reshape((-1, 1))
    # foreground_labels = np.array([1 for i in range(foreground_data.shape[0])]).reshape(-1, 1)
    # training_data = np.vstack([background_data,foreground_data])
    # training_labels = np.vstack([background_labels,foreground_labels]).ravel()
    #
    # # RGB for testing data
    # testing_data = im.reshape((-1, 3))
    #
    # # Convert to grayscale
    # gray_img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    # histSize = 256
    # histRange = (0, 256)  # the upper boundary is exclusive
    # accumulate = False
    # mask = im[background_ == 255]
    # hist = cv2.calcHist(gray_img, [0], mask, [histSize], histRange, accumulate=accumulate)
    # hist_normal = cv2.normalize(hist, hist, alpha=0, beta=gray_img.shape[0], norm_type=cv.NORM_MINMAX)
    # print(hist_normal)

    # Bayes
    # clf = GaussianNB(var_smoothing=1e-15)
    # clf.fit(training_data, training_labels)
    # probabilities = clf.predict_proba(testing_data)

    # Seperate fg/bg probababilities
    # prob_bg_raw = probabilities[:,0].reshape(m,n)
    # prob_fg_raw = probabilities[:,1].reshape(m,n)
    # Ibmean = mean(cv2.calcHist([im[background_ == 255]], [0], None, [256], [0, 256]))
    # Ifmean = mean(cv2.calcHist([im[foreground_ == 255]], [0], None, [256], [0, 256]))  # Taking the mean of the histogram
    # prob_fg = -log(abs(im - Ifmean) / (abs(im - Ifmean) + abs(im - Ibmean)))
    # prob_bg = -log(abs(im - Ibmean) / (abs(im - Ifmean) + abs(im - Ibmean)))
    # print(prob_fg.max(),prob_fg.min(),prob_fg.shape)
    # prob_bg = prob_bg_raw / (prob_bg_raw + prob_fg_raw)
    # prob_fg = prob_fg_raw / (prob_bg_raw + prob_fg_raw)
    # cv2.imwrite("prob_bg.png",prob_bg*255)
    # cv2.imwrite("prob_fg.png", prob_fg*255)

    # create graph
    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes((m,n))

    # add terminal edge weights
    print(foreground_.shape)
    background_edge_wt_matrix = -1 * lam * log(Fprob)
    source_edge_wt_matrix = -1 * lam * log(Bprop)

    #i,B for foreground
    background_edge_wt_matrix[foreground_ == 255] = 0
    #i,S for foreground
    source_edge_wt_matrix[foreground_ == 255] = inf

    # i,B for foreground
    background_edge_wt_matrix[background_ == 255] = inf
    # i,S for foreground
    source_edge_wt_matrix[background_ == 255] = 0

    g.add_grid_tedges(nodeids, source_edge_wt_matrix, background_edge_wt_matrix)
    # create and inter-node edge weights (left,right,up,down)
    gray_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    left = cv2.filter2D(gray_im, cv2.CV_32F, kernel=np.array([[0,0,0],[-1, 1, 0],[0,0,0]],dtype=np.float32))
    left_edge_wt_ = lam * exp(-1.0 * (abs(left) ** 2) / (2*sigma**2))
    structure = np.array([[0, 0, 0],
                          [1, 0, 0],
                          [0, 0, 0]
                          ])
    g.add_grid_edges(nodeids, weights=left_edge_wt_, structure=structure, symmetric=True)

    right = cv2.filter2D(gray_im,cv2.CV_32F,kernel=np.array([[0,0,0],[0, -1, 1],[0,0,0]],dtype=np.float32))
    right_edge_wt_ = lam * exp(-1.0 * (abs(right) ** 2) / (2*sigma**2))
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]
                          ])
    g.add_grid_edges(nodeids, weights=right_edge_wt_, structure=structure, symmetric=True)

    up = cv2.filter2D(gray_im,cv2.CV_32F,kernel=np.array([[0,-1,0],[0, 1, 0],[0,0,0]],dtype=np.float32))
    up_edge_wt_ = lam * exp(-1.0 * (abs(up) ** 2) / (2*sigma**2))
    structure = np.array([[0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0]
                          ])
    g.add_grid_edges(nodeids, weights=up_edge_wt_, structure=structure, symmetric=True)

    down = cv2.filter2D(gray_im, cv2.CV_32F, kernel=np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]],dtype=np.float32))
    down_edge_wt_ = lam * exp(-1.0 * (abs(down) ** 2) / (2*sigma**2))
    structure = np.array([[0, 0, 0],
                          [0, 0, 0],
                          [0, 1, 0]
                          ])
    g.add_grid_edges(nodeids, weights=down_edge_wt_, structure=structure, symmetric=True)
    print("Done Building Graph")
    return nodeids, g


def cut_graph(nodeids, g, image):
    """    Solve max flow of graph gr and return binary
        labels of the resulting segmentation."""

    # Find the maximum flow.
    g.maxflow()
    # Get the segments of the nodes in the grid.
    sgm = g.get_grid_segments(nodeids)

    return np.logical_not(sgm)
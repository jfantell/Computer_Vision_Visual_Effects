import numpy as np
import cv2
from scipy.sparse.linalg import spsolve, lsqr
from scipy.sparse import lil_matrix
from scipy.sparse.csc import csc_matrix
from scipy import sparse
import time
from utilities import fill_in_A



def blend(v,T,White_Pixels):
	start = time.time()
	I = T.copy().astype(np.uint8)
	for idx, (r,c) in enumerate(White_Pixels):
		I[r,c] = v[idx]
	end = time.time()
	print("Time To Construct I: {}".format(end-start))
	return I

def solve(A,b):
	start = time.time()
	v = spsolve(A,b)
	v[v > 255] = 255
	v[v < 0] = 0
	end = time.time()
	print("Time To Construct v: {}".format(end-start))
	return v

def make_composite_driver(S,T,M,A,White_Pixels):
	# Create b matrix
	b = fill_in_B(S,T,M,White_Pixels)
	
	# Solve linear equation
	v = solve(A,b)
	
	I = blend(v,T,White_Pixels)
	return I


def make_poisson_composite(S,T,M,output_dir):
	# TESTING
	# S = np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,8,9,0,0],[0,0,1,0,7,10,0,0],[0,0,2,5,6,0,0,0],[0,0,3,4,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],dtype=np.float32)
	# T = np.array([[125,125,125,125,125,125,125,125],[125,125,125,125,125,125,125,125],[125,125,125,125,125,125,125,125],[125,125,125,125,125,125,125,125],[125,125,125,125,125,125,125,125],[125,125,125,125,125,125,125,125],[125,125,125,125,125,125,125,125],[125,125,125,125,125,125,125,125]],dtype=np.float32)
	# M = np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,255,255,0,0],[0,0,255,0,255,255,0,0],[0,0,255,255,255,0,0,0],[0,0,255,255,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]],dtype=np.float32)

	# Compute poisson one channel at a time
	# store results of each channel
	print("\tMixed Composite\n\n")
	channel_composites = []
	
	# Binarize M
	ret,M = cv2.threshold(M,0,255,cv2.THRESH_BINARY)
	kernel = np.ones((9,9),np.uint8)
	M = cv2.erode(M,kernel,iterations=1)
	R, C = np.where(M==255)

	# Count number of white pixels
	White_Pixels = list(zip(R,C))
	N = len(White_Pixels)

	# Create A matrix
	A = fill_in_A(White_Pixels)
	
	# Run the Poission algorithm
	for channel in range(S.shape[2]):
		composite_image = make_composite_driver(S[:,:,channel],T[:,:,channel],M,A,White_Pixels)
		channel_composites.append(composite_image)

	# Merge all channels
	composite_image = cv2.merge(channel_composites)
	return composite_image

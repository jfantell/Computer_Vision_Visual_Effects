import numpy as np
import cv2
from scipy.sparse.linalg import spsolve, lsqr
from scipy.sparse import lil_matrix
from scipy.sparse.csc import csc_matrix
from scipy import sparse
import time

def fill_in_A(White_Pixels):
	start = time.time()
	# Initialize A array
	N = len(White_Pixels)

	# # Convert to sparse matrix
	A = lil_matrix((N,N),dtype=np.float32)
	A.setdiag(-4)

	'''
	Create dictionary where keys are white pixels (tuples)
	and values are indices in the White Pixels list
	'''
	coord_to_ind = {}
	for idx, (r,c) in enumerate(White_Pixels):
		coord_to_ind[(r,c)] = idx

	print("Number of white pixels {}".format(len(White_Pixels)))
	for A_row_idx, (r,c) in enumerate(White_Pixels):
		adjacent_pixels_list = [(r,c-1),(r+1,c),(r,c+1),(r-1,c)]
		for adjacent_pixel in adjacent_pixels_list:
			if adjacent_pixel in coord_to_ind:
				A_col_idx = coord_to_ind[adjacent_pixel]
				A[A_row_idx,A_col_idx] = 1

	# Convert lil to CSC format
	A = csc_matrix(A)

	end = time.time()
	print("Time To Construct A: {}".format(end-start))
	return A

def fill_in_B(S,M,White_Pixels):
	start = time.time()
	N = len(White_Pixels)
	b = np.zeros((N,), dtype=np.float32)

	S = S.copy()
	S[M==255] = 0

	kernel_laplacian = np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=np.float32)
	del_S = cv2.filter2D(S,cv2.CV_32F,kernel_laplacian)

	for idx, (r,c) in enumerate(White_Pixels):
		b[idx] = 0 - del_S[r,c]
	end = time.time()
	print("Time To Construct B: {}".format(end-start))
	return b

def blend(v,S,White_Pixels):
	start = time.time()
	I = S.copy().astype(np.uint8)
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

# def translate(S,T, M):
# 	offset = (0, 0)
# 	TM = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
# 	S = cv2.warpAffine(S, TM, (T.shape[1], T.shape[0]))
# 	M = cv2.warpAffine(M, TM, (T.shape[1], T.shape[0]))
# 	return S, M

def make_composite_driver(S,M,A,White_Pixels):
	# Create b matrix
	b = fill_in_B(S,M,White_Pixels)
	
	# Solve linear equation
	v = solve(A,b)
	
	I = blend(v,S,White_Pixels)
	return I


def make_poisson_composite(S,M,output_dir):
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
	kernel = np.ones((3,3),np.uint8)
	M = cv2.erode(M,kernel,iterations=1)

	# Save mask
	cv2.imwrite(f'{output_dir}/mask.jpg',M)

	R, C = np.where(M==255)

	# Count number of white pixels
	White_Pixels = list(zip(R,C))
	N = len(White_Pixels)

	# Create A matrix
	A = fill_in_A(White_Pixels)
	
	# Run the Poission algorithm
	for channel in range(S.shape[2]):
		composite_image = make_composite_driver(S[:,:,channel],M,A,White_Pixels)
		channel_composites.append(composite_image)

	# Merge all channels
	composite_image = cv2.merge(channel_composites)
	return composite_image

import numpy as np
import cv2

'''
𝐶𝐿(𝑥,𝑦)=𝐷[(𝑥−1,𝑦),(𝑥+1,𝑦)]+𝐷[(𝑥,𝑦−1),(𝑥−1,𝑦)]
𝐶𝑈(𝑥,𝑦)=𝐷[(𝑥−1,𝑦),(𝑥+1,𝑦)]
𝐶𝑅(𝑥,𝑦)=𝐷[(𝑥−1,𝑦)(𝑥+1,𝑦)]+𝐷[(𝑥,𝑦−1),(𝑥+1,𝑦)]
'''
def draw_min_seam(img,min_seam):
	drawn_img = img.copy()
	M = drawn_img.shape[0]
	for i in range(M):
		drawn_img[i,min_seam[i]] = (255,0,255)
	return drawn_img.astype(np.uint8)

def remove_seam(img,min_seam):
	M = img.shape[0]
	N = img.shape[1]
	resized_img = np.zeros((M,N-1,3))
	for i in range(M):
		resized_img[i,:min_seam[i]] = img[i,:min_seam[i]] 
		resized_img[i,min_seam[i]:] = img[i,min_seam[i]+1:]
	return resized_img.astype(np.uint8)

def diff(arr1,arr2):
	#return np.absolute(arr1) - np.absolute(arr2)
	#return np.absolute(arr2) - np.absolute(arr1)
	return np.absolute(arr1-arr2)

def compute_costs(img):
	M = img.shape[0]
	N = img.shape[1]

	# Three cost functions for each pixel
	CL = np.zeros((M,N))
	CU = np.zeros((M,N))
	CR = np.zeros((M,N))

	# Create three shifted views of image
	shift_left = img[1:,:-2]
	shift_right = img[1:,2:]
	shift_top = img[:-1,1:-1]

	# Compute costs
	CL[1:,1:-1] = diff(shift_right,shift_left) + diff(shift_left,shift_top)
	CU[1:,1:-1] = diff(shift_right,shift_left)
	CR[1:,1:-1] = diff(shift_right,shift_left) + diff(shift_right,shift_top)

	# Top row (0s for right and left)
	CU[0,1:-1] = diff(shift_right[0,:],shift_left[0,:])

	# Left and right columns
	CL[:,0] = 10**10; CL[:,-1] = 10**10
	CU[:,0] = 10**10; CU[:,-1] = 10**10
	CR[:,0] = 10**10; CR[:,-1] = 10**10

	return CL, CU, CR

def compute_min_costs(CL, CU, CR):
	M = CL.shape[0]
	N = CL.shape[1]

	# Min cost matrix
	MC = CU
	for i in range(1,M):
		M_TL = MC[i-1,:-2] + CL[i,1:-1]
		M_T = MC[i-1,1:-1] + CU[i,1:-1]
		M_TR = MC[i-1,2:] + CR[i,1:-1]
		MC[i,1:-1]  = np.minimum(M_TL,np.minimum(M_T,M_TR))
	return MC

def find_min_seam(min_costs):
	M = min_costs.shape[0]
	N = min_costs.shape[1]
	min_seam = []
	column_index = np.argmin(min_costs[-1,:])
	min_seam.append(column_index)
	for i in range(M-2,-1,-1):
		tmp_min_cost = np.argmin([min_costs[i,column_index],min_costs[i,column_index-1],min_costs[i,column_index+1]])
		if(tmp_min_cost == 1):
			column_index-= 1
		elif(tmp_min_cost == 2):
			column_index+=1
		min_seam.append(column_index)
	min_seam = min_seam[::-1]
	return min_seam

def main():
	# img = cv2.imread("geese.jpg")
	img = cv2.imread("arch-seam.jpg")
	while img.shape[0] != img.shape[1]:
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img_gray = img_gray.astype(np.float64)
		(CL, CU, CR) = compute_costs(img_gray)
		min_costs = compute_min_costs(CL, CU, CR)
		min_seam = find_min_seam(min_costs)
		drawn_img = draw_min_seam(img,min_seam)
		resized_img = remove_seam(img,min_seam)
		if(abs(img.shape[0]-img.shape[1]) == 1):
			cv2.imwrite("resized_img.png",resized_img)
			cv2.imwrite("drawn_img.png",drawn_img)
		img = resized_img

if __name__ == '__main__':
	main()


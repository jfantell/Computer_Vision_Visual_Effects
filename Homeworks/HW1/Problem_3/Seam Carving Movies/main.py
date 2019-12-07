import numpy as np
import cv2

MASK_LEFT = 400
MASK_RIGHT = 1200
BLENDING_OFFSET = 0

def build_composite(img_left,img_right,min_seam,count):
	M = img_left.shape[0]
	N = img_right.shape[1]
	composite_img = np.zeros((M,N,3))
	for i in range(M):
		composite_img[i,:min_seam[i]-BLENDING_OFFSET] = img_left[i,:min_seam[i]-BLENDING_OFFSET]
		blended_img = (0.5 * img_left[i,min_seam[i]-BLENDING_OFFSET:min_seam[i]+BLENDING_OFFSET]) + 0.5 * (img_right[i,min_seam[i]-BLENDING_OFFSET:min_seam[i]+BLENDING_OFFSET])
		composite_img[i,min_seam[i]-BLENDING_OFFSET:min_seam[i]+BLENDING_OFFSET] = blended_img
		composite_img[i,min_seam[i]+BLENDING_OFFSET:] = img_right[i,min_seam[i]+BLENDING_OFFSET:]
		# composite_img[i,min_seam[i]] = (255,0,0)
	return composite_img.astype(np.uint8)

def diff(arr1,arr2):
	diff = np.abs(arr2-arr1)
	diff = cv2.GaussianBlur(diff,(5,5),0)
	return diff

# Dynamic recursion to calculate pixel costs
def create_views(left,right):
	M = left.shape[0]
	N = left.shape[1]

	# Three cost functions for each pixel
	CL = np.zeros((M,N),dtype=np.float64)
	CU = np.zeros((M,N),dtype=np.float64)
	CR = np.zeros((M,N),dtype=np.float64)

	# Create three shifted views of image
	shift_left_leftv = left[1:,:-2]
	shift_right_leftv = left[1:,2:]
	shift_right_rightv = right[1:,2:]
	shift_top_rightv = right[:-1,1:-1]
	shift_top_leftv = left[:-1,1:-1]

	# Compute costs for each view
	CL[1:,1:-1] = diff(shift_right_rightv,shift_left_leftv) + diff(shift_top_rightv,shift_left_leftv)
	CU[1:,1:-1] = diff(shift_right_rightv,shift_left_leftv)
	CR[1:,1:-1] = diff(shift_right_rightv,shift_left_leftv) + diff(shift_top_leftv,shift_right_rightv)

	# Top row base case (there is no top above the top row)
	# CU[0,1:-1] = diff(shift_right_rightv[0,:],shift_left_leftv[0,:])
	tmp = diff(shift_right_rightv,shift_left_leftv)
	CU[0,1:-1] = tmp[0,:]

	CU[:,0:MASK_LEFT] = 10**10
	CU[:,MASK_RIGHT:N] = 10**10
	CR[:,0:MASK_LEFT] = 10**10
	CR[:,MASK_RIGHT:N] = 10**10
	CL[:,0:MASK_LEFT] = 10**10
	CL[:,MASK_RIGHT:N] = 10**10

	# Left and right columns
	return CL, CU, CR

# Compute cumulative pixel costs
def compute_min_costs(CL, CU, CR, count):
	M = CU.shape[0]
	N = CU.shape[1]

	# Extract non-mask part of matrix
	CL_ = CL[:,MASK_LEFT-1:MASK_RIGHT+1]
	CU_ = CU[:,MASK_LEFT-1:MASK_RIGHT+1]
	CR_ = CR[:,MASK_LEFT-1:MASK_RIGHT+1]

	# Min cost matrix
	MC_ = CU_
	for i in range(1,M):
		M_TL_ = MC_[i-1,:-2] + CL_[i,1:-1]
		M_T_ = MC_[i-1,1:-1] + CU_[i,1:-1]
		M_TR_ = MC_[i-1,2:] + CR_[i,1:-1]
		MC_[i,1:-1]  = np.minimum(M_TL_,np.minimum(M_T_,M_TR_))

	MC = CU
	MC[:,MASK_LEFT-1:MASK_RIGHT+1] = MC_
	return MC

# Determine indices of best seam
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
	# Get movie file paths
	left_video = "./inputs/LEFT_VIDEO.mp4"
	right_video = "./inputs/RIGHT_VIDEO.MOV"

	# Open video streams
	cap_left = cv2.VideoCapture(left_video)
	cap_right = cv2.VideoCapture(right_video)
	start_frame_number = 400
	cap_left.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
	cap_right.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
	fps_left = cap_left.get(cv2.CAP_PROP_FPS)
	fps_right = cap_right.get(cv2.CAP_PROP_FPS)
	assert(fps_left == fps_right)
	print("Video FPS: {}".format(fps_left))

	# Setup video writer
	# Source: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
	frame_width = int(cap_left.get(3))
	frame_height = int(cap_left.get(4))
	writer = cv2.VideoWriter('./outputs/twinning.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps_left, (frame_width,frame_height))

	count = 0
	while cap_left.isOpened():
		ret_left, current_left = cap_left.read()
		ret_right, current_right = cap_right.read()
		if(ret_left == True and ret_right == True):
			# Convert frames to grayscale
			current_left_gray = cv2.cvtColor(current_left, cv2.COLOR_BGR2GRAY)
			current_right_gray = cv2.cvtColor(current_right, cv2.COLOR_BGR2GRAY)
			# Convert to float64
			current_left_gray = current_left_gray.astype(np.float64)
			current_right_gray = current_right_gray.astype(np.float64)

			cv2.imshow('DIFF',diff(current_left_gray,current_right_gray).astype(np.uint8))
					
			# Compute cost matrices
			(CL, CU, CR) = create_views(current_left_gray,current_right_gray)
			
			# Compute minimum cost matrix according to algorithm proposed by pap[er]
			min_costs = compute_min_costs(CL, CU, CR,count)
			
			# Traverse the min cost matrix in reverse
			min_seam = find_min_seam(min_costs)
			diff_frame = diff(current_left_gray,current_right_gray).astype(np.uint8)
			composited_video_frame = build_composite(current_left,current_right,min_seam,count)
			cv2.imshow('Frame',composited_video_frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
			writer.write(composited_video_frame)
		else:
			break
		count += 1

	# Release video capture and writer objects
	cap_left.release()
	cap_right.release()
	writer.release()


if __name__ == '__main__':
	main()
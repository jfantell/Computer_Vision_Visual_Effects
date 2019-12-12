import cv2
import numpy as np
import sys
import os
from utilities import *

def normalize(x):
	return (x - x.min()) / (x.max() - x.min())

def clip(x):
	x[x>1] = 1
	x[x<0] = 0
	return x

def find_alpha(I):
	# normalize I and split channels
	I_n = normalize(I)
	I_b, I_g, I_r = cv2.split(I_n)

	# choose heuristics for a1 and a2
	a1 = 5
	a2 = 1

	# compute alpha according to equation 2.4
	a = 1 - a1 * (I_g - a2 * I_r)
	a = clip(a)
	a = np.expand_dims(a, axis=2)
	return a

def main():
	input_dir_base = './inputs'
	output_dir_base = './outputs'
	for sub_dir in os.listdir(input_dir_base):
		input_dir = f"{input_dir_base}/{sub_dir}"
		if(os.path.isdir(input_dir)):
			# Create output directory
			output_dir = f"{output_dir_base}/{sub_dir}"
			make_dir(output_dir)
			
			# Get fg and bg file paths
			F = get_file_path(input_dir,"foreground")
			B = get_file_path(input_dir,"background")

			# Open fg and bg images
			F = cv2.imread(F,cv2.IMREAD_COLOR).astype(np.float32)
			B = cv2.imread(B,cv2.IMREAD_COLOR).astype(np.float32)

			# Find alpha
			a = find_alpha(F)

			# Construct matte
			I = a * F + (1-a) * B
			cv2.imwrite(f"{output_dir}/composite_img.png",I)

if __name__ == '__main__':
	main()
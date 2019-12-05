from utilities import *
import composite_mixed
import cv2
import os
import sys

def main():
	# Read images
	input_dir_base = './inputs'
	output_dir_base = './outputs'
	for subdir in list(os.walk(input_dir_base))[0][1]:
		if(True):
			# Create output directory
			output_dir = f"{output_dir_base}/{subdir}"
			make_dir(output_dir)
			
			# Determine image file extension
			input_dir = f"{input_dir_base}/{subdir}"
			
			# Read images
			S_path = get_file_path(input_dir,"source")
			M_path = get_file_path(input_dir,"mask")
			S = cv2.imread(S_path)
			M = cv2.imread(M_path,0) # Grayscale

			# Check to confirm all same dimensions
			if(S.shape[:2] != M.shape[:2]):
				print("Images should all have same dimensions")
				return
			else:
				print("Original Image Dimensions: S {} M {}".format(S.shape,M.shape))
			
			# Resize images
			fx = 1; fy = 1
			S = cv2.resize(S,(0,0),fx=fx,fy=fy)
			M = cv2.resize(M,(0,0),fx=fx,fy=fy)
			print("Resized Image Dimensions: S {} M {}".format(S.shape,M.shape))

			# Create composite mixed gradients image
			composite_image_mixed = composite_mixed.make_poisson_composite(S,M,output_dir)
			cv2.imwrite(f"{output_dir}/composite_image_mixed.png",composite_image_mixed)
#            sys.exit()

if __name__ == '__main__':
	main()

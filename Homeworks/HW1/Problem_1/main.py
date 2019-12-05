from utilities import *
import composite
import composite_mixed
import cv2
import os
import sys

def main():
	# Read images
	input_dir_base = './inputs'
	output_dir_base = './outputs'
	for subdir in list(os.walk(input_dir_base))[0][1]:
		if(subdir!='3'):
			# Create output directory
			output_dir = f"{output_dir_base}/{subdir}"
			make_dir(output_dir)
			
			# Determine image file extension
			input_dir = f"{input_dir_base}/{subdir}"
			
			# Read images
			S_path = get_file_path(input_dir,"source")
			T_path = get_file_path(input_dir,"target")
			M_path = get_file_path(input_dir,"mask")
			S = cv2.imread(S_path)
			T = cv2.imread(T_path)
			M = cv2.imread(M_path,0) # Grayscale

			# Check to confirm all same dimensions
			if not( (S.shape[:2] == T.shape[:2]) and (S.shape[:2] == M.shape[:2]) ):
				print("Images should all have same dimensions")
				return
			else:
				print("Original Image Dimensions: S {} T {} M {}".format(S.shape,T.shape,M.shape))
			
			# Resize images
			fx = 1; fy = 1
			S = cv2.resize(S,(0,0),fx=fx,fy=fy)
			T = cv2.resize(T,(0,0),fx=fx,fy=fy)
			M = cv2.resize(M,(0,0),fx=fx,fy=fy)
			print("Resized Image Dimensions: S {} T {} M {}".format(S.shape,T.shape,M.shape))
			
			# Create composite image
			composite_image = composite.make_poisson_composite(S,T,M,output_dir)
			cv2.imwrite(f"{output_dir}/composite_image.png",composite_image)

			# Create composite mixed gradients image
			composite_image_mixed = composite_mixed.make_poisson_composite(S,T,M,output_dir)
			cv2.imwrite(f"{output_dir}/composite_image_mixed.png",composite_image_mixed)
#            sys.exit()

if __name__ == '__main__':
	main()

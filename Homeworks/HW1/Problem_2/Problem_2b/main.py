from utilities import *
import composite_mixed_video
import cv2
import os
import sys

def main():
	# Read images
	mask_files_dir = './video_input/mask_frames'
	source_files_dir = './video_input/source_frames'
	output_path= './poisson_inpainting.avi'
	mask_files = [file for file in os.listdir(mask_files_dir) if file.endswith((".png",".jpg"))]
	source_files = [file for file in os.listdir(source_files_dir) if file.endswith((".png",".jpg"))]
	mask_files.sort()
	source_files.sort()

	# Open first file to get frame specs
	print("{}/{}".format(mask_files_dir,mask_files[0]))
	img = cv2.imread("{}/{}".format(mask_files_dir,mask_files[0]))
	M = img.shape[0]; N = img.shape[1]

	writer = cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc('M','J','P','G'), 20, (N,M))
	for mask_file in mask_files:
		#Get corresponding source file
		print(mask_file)
		M = cv2.imread("{}/{}".format(mask_files_dir,mask_file),0)
		S = cv2.imread("{}/{}".format(source_files_dir,mask_file)) # Grayscale

		# Check to confirm all same dimensions
		if(S.shape[:2] != M.shape[:2]):
			print("Images should all have same dimensions")
			return
		else:
			print("Original Image Dimensions: S {} M {}".format(S.shape,M.shape))
		
		# Create composite mixed gradients image
		composite_image_mixed = composite_mixed_video.make_poisson_composite(S,M)
		writer.write(composite_image_mixed)
	writer.release()

if __name__ == '__main__':
	main()

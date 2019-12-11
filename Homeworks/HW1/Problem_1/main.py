from utilities import *
import composite
import cv2
import os
import sys

def main():
	# Read images
	input_dir_base = './inputs'
	output_dir_base = './outputs'
	for subdir in list(os.walk(input_dir_base))[0][1]:
		if(subdir in ['1','11']):
			# Create output directory
			output_dir = f"{output_dir_base}/{subdir}"
			make_dir(output_dir)
			
			# Determine image file extension
			input_dir = f"{input_dir_base}/{subdir}"
			
			# Read images
			retS, S_path = get_file_path(input_dir,"source")
			retT, T_path = get_file_path(input_dir,"target")
			retM, M_path = get_file_path(input_dir,"mask")
			S = cv2.imread(S_path)
			M = cv2.imread(M_path,0) # Grayscale
			# Target (will define later)
			T = None

			# Used when Target is a movie
			cap_target = None
			composite_writer = None
			mixed_composite_writer = None

			# This means the target is actually a movie
			# instead of an image
			if retT == 2:
				cap_target = cv2.VideoCapture(T_path)
				start_frame_number = 0
				cap_target.set(cv2.CAP_PROP_POS_FRAMES,start_frame_number)
				fps = cap_target.get(cv2.CAP_PROP_FPS)
				frame_width = int(cap_target.get(3))
				frame_height = int(cap_target.get(4))
				composite_writer = cv2.VideoWriter(f"{output_dir}/composite.MOV",cv2.VideoWriter_fourcc(*"MJPG"),fps,(frame_width,frame_height))
				mixed_composite_writer = cv2.VideoWriter(f"{output_dir}/mixed_composite.MOV",cv2.VideoWriter_fourcc(*"XVID"),fps,(frame_width,frame_height))
			else:
				T = cv2.imread(T_path)
			
			if retT == 2:
				ret, T = cap_target.read()
				cap_target.set(cv2.CAP_PROP_POS_FRAMES,start_frame_number) #reset video capture object

			if (T is None) or (S is None) or (M is None): #ensure all files exist
				print("Error reading at least one of the input files")
			elif not( (S.shape[:2] == T.shape[:2]) and (S.shape[:2] == M.shape[:2]) ): # ensure all same dimension
					print("Images should all have same dimensions")
					return
			else:
				print("Original Image Dimensions: S {} T {} M {}".format(S.shape,T.shape,M.shape))
			
			if retT == 1: # Image pipeline
				fx = 1; fy = 1
				S = cv2.resize(S,(0,0),fx=fx,fy=fy)
				T = cv2.resize(T,(0,0),fx=fx,fy=fy)
				M = cv2.resize(M,(0,0),fx=fx,fy=fy)
				print("Resized Image Dimensions: S {} T {} M {}".format(S.shape,T.shape,M.shape))

				source_intensities_3ch, source_coordinates = composite.solve_possion(S,T,M,output_dir,mode="Regular")
				# source_intensities_mg_3ch, _ = composite.solve_possion(S,T,M,output_dir,mode="Mixed_Gradients")

				composite_image = composite.blend(source_intensities_3ch,T,source_coordinates)
				# composite_image_mixed = composite.blend(source_intensities_mg_3ch,T,source_coordinates)

				cv2.imwrite(f"{output_dir}/composite_image.png",composite_image)
				# cv2.imwrite(f"{output_dir}/composite_image_mixed.png",composite_image_mixed)
			elif retT == 2: # Video pipeline
				# Poisson Intensities
				source_intensities_3ch = None
				# Mixed Gradient Poisson Intensities
				source_intensities_mg_3ch = None
				# Source pixel coordinates
				source_coordinates = None
				count = 0

				while cap_target.isOpened():
					ret, T = cap_target.read()
					if count == 0:
						source_intensities_3ch, source_coordinates = composite.solve_possion(S,T,M,output_dir,mode="Regular")
						source_intensities_mg_3ch, _ = composite.solve_possion(S,T,M,output_dir,mode="Mixed_Gradients")

					composite_image = composite.blend(source_intensities_3ch,T,source_coordinates)
					composite_image_mixed = composite.blend(source_intensities_mg_3ch,T,source_coordinates)

					cv2.imshow('Composite Regular',composite_image)	
					key = cv2.waitKey(1) & 0xFF
					if key == ord('q'):
						break

					composite_writer.write(composite_image)
					mixed_composite_writer.write(composite_image_mixed)

					count += 1
			if retT == 2:
				cap_target.release()
				composite_writer.release()
				mixed_composite_writer.release()

if __name__ == '__main__':
	main()

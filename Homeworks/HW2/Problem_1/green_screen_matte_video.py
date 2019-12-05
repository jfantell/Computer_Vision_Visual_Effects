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
	a1 = 7
	a2 = 1

	# compute alpha according to equation 2.4
	a = 1 - a1 * (I_g - a2 * I_r)
	a = clip(a)
	a = np.expand_dims(a, axis=2)
	return a

def main():
	# Read images
	foreground_video_path = './video_input/problem_1c_foreground.mp4'
	composite_video_path = './video_input/problem_1c_composite.avi'
	background_image_path = './video_input/problem_1c_background.png'

	#  Open video streams
	foreground_video = cv2.VideoCapture(foreground_video_path)
	start_frame_number = 0
	foreground_video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
	fps = foreground_video.get(cv2.CAP_PROP_FPS)
	print("Video FPS: {}".format(fps))

	# Setup video writer
	# Source: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
	frame_width = int(foreground_video.get(3))
	frame_height = int(foreground_video.get(4))
	writer = cv2.VideoWriter(composite_video_path,cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

	# Open background image
	B = cv2.imread(background_image_path).astype(np.float32)

	while foreground_video.isOpened():
		ret, F = foreground_video.read()
		if(F!=None):
			# Find alpha
			a = find_alpha(F)

			# Construct matte
			I = a * F + (1-a) * B
			writer.write(I.astype(np.uint8))
	
	foreground_video.release()
	writer.release()

if __name__ == '__main__':
	main()
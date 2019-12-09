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
	I = I.astype(np.float32)
	I_n = normalize(I)
	I_b, I_g, I_r = cv2.split(I_n)

	# choose heuristics for a1 and a2
	a1 =7
	a2 = 1

	# compute alpha according to equation 2.4
	a = 1 - a1 * (I_g - a2 * I_r)
	a = clip(a)
	a = np.expand_dims(a, axis=2)
	return a

def main():
	# Read videos
	scene = "text_scene"
	foreground_video_cap_path = './video_input/{}.MOV'.format(scene)
	background_video_cap_path = './video_input/lions.mp4'

	#  Open video streams
	foreground_video_cap = cv2.VideoCapture(foreground_video_cap_path)
	background_video_cap = cv2.VideoCapture(background_video_cap_path)
	start_frame_number = 0
	foreground_video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
	background_video_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
	fps_foreground = foreground_video_cap.get(cv2.CAP_PROP_FPS)
	fps_background = background_video_cap.get(cv2.CAP_PROP_FPS)
	print(fps_foreground,fps_background)
	assert(fps_foreground == fps_background)
	print("Video FPS: {}".format(fps_foreground))

	# Setup video writer
	# Source: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
	frame_width = int(foreground_video_cap.get(3))
	frame_height = int(foreground_video_cap.get(4))
	composite_video_path = './video_output/{}.MOV'.format(scene)
	writer = cv2.VideoWriter(composite_video_path,cv2.VideoWriter_fourcc(*'XVID'), fps_foreground, (frame_width,frame_height))

	while foreground_video_cap.isOpened():
		ret1, F = foreground_video_cap.read()
		ret2, B = background_video_cap.read()
		if(F is not None and B is not None):
			# Find alpha
			a = find_alpha(F)

			# Construct matte
			I = a * F.astype(np.float32) + (1-a) * B.astype(np.float32)

			cv2.imshow('Final Frame', I.astype(np.uint8))
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break

			writer.write(I.astype(np.uint8))
		else:
			break
	
	foreground_video_cap.release()
	writer.release()

if __name__ == '__main__':
	main()
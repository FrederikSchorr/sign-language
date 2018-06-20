# import the necessary packages
import time
import os
import sys

import numpy as np
import pandas as pd

import cv2


from frame import images_normalize
from videocapture import camera_resolution, frame_show, video_show, video_capture
from opticalflow import frames2flows
from datagenerator import VideoClasses
from feature_2D import features_2D_load_model
from train_mobile_lstm import lstm_load
from predict_mobile_lstm import predict


def main():
	
	# important constants
	nClasses = 21   	# number of classes
	nFramesNorm = 20    # number of frames per video
	nMinDim = 288   	# smaller dimension of saved video-frames. typically same as video 


	# feature extractor 
	diFeature = {"sName" : "mobilenet",
		"tuInputShape" : (224, 224, 3),
		"tuOutputShape" : (1024, )
	}

	# files
	sClassFile	= "data-set/01-ledasila/%03d/class.csv"%(nClasses)
	sModelFile 	= "model/20180620-1701-ledasila021-oflow-mobile-lstm-best.h5"

	print("\nStarting gesture recognition live demo from " + os.getcwd())
	
	# load label description
	oClasses = VideoClasses(sClassFile)

	# Load pretrained MobileNet model without top layer 
	keFeature = features_2D_load_model(diFeature)
	h, w, c = diFeature["tuInputShape"]

	# Load trained LSTM network
	keLSTM = lstm_load(sModelFile, nFramesNorm, diFeature["tuOutputShape"][0], oClasses.nClasses)

	# open a pointer to the webcam video stream
	oStream = cv2.VideoCapture(0)
	camera_resolution(oStream, 352, 288)

	# loop over action states
	print("Launch video capture screen ...")
	sResults = ""
	while True:
		# show live video and wait for key stroke
		key = video_show(oStream, "green", "Press <blank> to start", sResults)
		
		# start!
		if key == ord(' '):
			# countdown n sec
			video_show(oStream, "orange", "Recording starts in ", None, 3)
			
			# record video for n sec
			fElapsed, arFrames = video_capture(oStream, "red", "Recording ", 5)
			print("\nCaptured video: {:.1f} sec, {} frames, {:.1f} fps". \
				format(fElapsed, len(arFrames), len(arFrames)/fElapsed))

			# show orange wait box
			frame_show(oStream, "orange", "Translating sign ...")

			# Translate frames to flows - these are already scaled between [-1.0, 1.0]
			print("Calculate optical flow ...")
			arFlows = frames2flows(arFrames, bThirdChannel = True)

			# downsample number of frames and crop to center
			arFlows = images_normalize(arFlows, nFramesNorm, h, w, bRescale = False)

			# Calculate feature from flows
			print("Predict %s features ..." % keFeature.name)
			arFeatures = keFeature.predict(arFlows, verbose=1)

			# Classify flows with LSTM network
			print("Final predict with LSTM ...")
			nLabel, sLabel, fProba = predict(arFeatures, keLSTM, oClasses, nTop = 3)

			sResults = "Identified sign: [%d] %s (confidence %.0f%%)" % (nLabel, sLabel, fProba*100.)
			print(sResults)

			# ready for next video	

		# quit
		elif key == ord('q'):
			break


	# do a bit of cleanup
	oStream.release()
	cv2.destroyAllWindows()

	return


if __name__ == '__main__':
	main()
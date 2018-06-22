# import the necessary packages
import time
import os
import glob
import sys
import random

import numpy as np
import pandas as pd

import cv2

from frame import video2frames, images_normalize, frames_downsample, images_crop, images_resize_aspectratio, frames_show, frames2files, files2frames
from videocapture import camera_resolution, frame_show, video_show, video_capture
from opticalflow import frames2flows, flows2colorimages, flows2file
from datagenerator import VideoClasses
from feature_2D import features_2D_load_model
from train_mobile_lstm import lstm_load
from predict_mobile_lstm import predict


def predict_frames(arFrames:np.array, keFeature, keLSTM, oClasses:VideoClasses) \
	-> (int, str, float):
	
	_, h, w, _ = keFeature.input_shape
	_, nFramesNorm, _ = keLSTM.input_shape

	# resize 
	arFrames = images_crop(images_resize_aspectratio(arFrames, min(h,w)), h, w)

	# Translate frames to flows - these are already scaled between [-1.0, 1.0]
	print("Calculate optical flow ...")
	arFlows = frames2flows(arFrames, bThirdChannel = True)
	frames_show(flows2colorimages(arFlows), 35)

	# downsample
	arFlows = frames_downsample(arFlows, nFramesNorm)

	# Calculate feature from flows
	print("Predict %s features ..." % keFeature.name)
	arFeatures = keFeature.predict(arFlows, verbose=1)

	# Classify flows with LSTM network
	print("Final predict with LSTM ...")
	nLabel, sLabel, fProba = predict(arFeatures, keLSTM, oClasses, nTop = 3)

	return nLabel, sLabel, fProba


def main():
	
	# important constants
	nClasses = 21   	# number of classes
	nFramesNorm = 20    # number of frames per video

	# feature extractor 
	diFeature = {"sName" : "mobilenet",
		"tuInputShape" : (224, 224, 3),
		"tuOutputShape" : (1024, )
	}

	# files
	sClassFile	= "data-set/01-ledasila/%03d/class.csv"%(nClasses)
	sVideoDir   = "data-set/01-ledasila/%03d"%(nClasses)
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
	oStream = cv2.VideoCapture(1) # 0 for the primary webcam
	camera_resolution(oStream, nWidth=320, nHeight=240)
	oStream.set(cv2.CAP_PROP_FPS, 25)

	h, w, _ = diFeature["tuInputShape"]
	liVideos = glob.glob(sVideoDir + "/train/*/*.*")
	nCount = 0
	sResults = ""
	# loop over action states
	print("Launch video capture screen ...")
	while True:
		# show live video and wait for key stroke
		key = video_show(oStream, "green", "Press <blank> to start", sResults)
		
		# start!
		if key == ord(' '):
			# countdown n sec
			video_show(oStream, "orange", "Recording starts in ", None, 3)
			
			# record video for n sec
			fElapsed, arFrames = video_capture(oStream, "red", "Recording ", 5)
			print("\nCaptured video: %.1f sec, %s, %.1f fps" % \
				(fElapsed, str(arFrames.shape), len(arFrames)/fElapsed))

			# show orange wait box
			frame_show(oStream, "orange", "Translating sign ...")

			# predict video
			nLabel, sLabel, fProba = predict_frames(arFrames, keFeature, keLSTM, oClasses)
			sResults = "Identified sign: [%d] %s (confidence %.0f%%)" % (nLabel, sLabel, fProba*100.)
			print(sResults)
			nCount += 1

			# ready for next video	

		# debug
		elif key == ord('d'):
			sVideoFile = random.choice(liVideos)
			arFrames = video2frames(sVideoFile, 256)
			print("DEBUG: Loaded %s | shape %s" % (sVideoFile, str(arFrames.shape)))
			frames_show(arFrames, 35)
			nLabel, sLabel, fProba = predict_frames(arFrames, keFeature, keLSTM, oClasses)
			sResults = "Identified sign: [%d] %s (confidence %.0f%%)" % (nLabel, sLabel, fProba*100.)

		# quit
		elif key == ord('q'):
			break


	# do a bit of cleanup
	oStream.release()
	cv2.destroyAllWindows()

	return


if __name__ == '__main__':
	main()
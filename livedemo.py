# import the necessary packages
import time
import os
import glob
import sys
import random

import numpy as np
import pandas as pd

import cv2

from timer import Timer
from frame import video2frames, images_normalize, frames_downsample, images_crop
from frame import images_resize_aspectratio, frames_show, frames2files, files2frames, video_length
from videocapture import video_start, frame_show, video_show, video_capture
from opticalflow import frames2flows, flows2colorimages, flows2file
from datagenerator import VideoClasses
from feature_2D import features_2D_load_model
from train_mobile_lstm import lstm_load
from predict_mobile_lstm import predict


def predict_frames(arFrames:np.array, keFeature, keLSTM, oClasses:VideoClasses) \
	-> (int, str, float):
	
	_, h, w, _ = keFeature.input_shape
	_, nFramesNorm, _ = keLSTM.input_shape

	timer = Timer()

	# resize 
	arFrames = images_crop(images_resize_aspectratio(arFrames, min(h,w)), h, w)

	# downsample
	arFrames = frames_downsample(arFrames, nFramesNorm)

	# Translate frames to flows - these are already scaled between [-1.0, 1.0]
	print("Calculate optical flow on %d frames ..." % len(arFrames))
	timer.start()
	arFlows = frames2flows(arFrames, bThirdChannel = True, bShow = True)
	print("Optical flow per frame: %.3f" % (timer.stop() / len(arFrames)))

	# Calculate feature from flows
	print("Predict features with %s..." % keFeature.name)
	arFeatures = keFeature.predict(arFlows, batch_size = nFramesNorm // 4, verbose=1)

	# Classify flows with LSTM network
	print("Final predict with LSTM ...")
	nLabel, sLabel, fProba = predict(arFeatures, keLSTM, oClasses, nTop = 3)

	return nLabel, sLabel, fProba


def predict_flows(arFlows:np.array, keFeature, keLSTM, oClasses:VideoClasses) \
	-> (int, str, float):
	
	_, h, w, _ = keFeature.input_shape
	_, nFramesNorm, _ = keLSTM.input_shape

	# resize 
	arFlows = images_crop(images_resize_aspectratio(arFlows, min(h,w)), h, w)

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
	
	# dataset
	diVideoSet = {"sName" : "04-chalearn",
		"nClasses" : 10,   # number of classes
		"nFramesNorm" : 40,    # number of frames per video
		"nMinDim" : 240,   # smaller dimension of saved video-frames
		"tuShape" : (240, 320), # height, width
		"nFpsAvg" : 10,
		"nFramesAvg" : 50, 
		"fDurationAvg" : 5.0} # seconds 

	# feature extractor 
	diFeature = {"sName" : "mobilenet",
		"tuInputShape" : (224, 224, 3),
		"tuOutputShape" : (1024, )
	}

	# files
	sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
	sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
	
	sModelFile      = "model/20180623-0429-04-chalearn010-otvl1-mobile-lstm-best.h5"


	print("\nStarting gesture recognition live demo ... ")
	print(os.getcwd())
	print(diVideoSet)
	
	# load label description
	oClasses = VideoClasses(sClassFile)

	# Load pretrained MobileNet model without top layer 
	keFeature = features_2D_load_model(diFeature)
	h, w, c = diFeature["tuInputShape"]

	# Load trained LSTM network
	keLSTM = lstm_load(sModelFile, diVideoSet["nFramesNorm"], diFeature["tuOutputShape"][0], oClasses.nClasses)

	# open a pointer to the webcam video stream
	oStream = video_start(device = 1, tuResolution = (320, 240), nFramePerSecond = diVideoSet["nFpsAvg"])

	liVideosDebug = glob.glob(sVideoDir + "/train/*/*.*")
	nCount = 0
	sResults = ""

	# loop over action states
	while True:
		# show live video and wait for key stroke
		key = video_show(oStream, "green", "Press <blank> to start", sResults, tuRectangle = (h, w))
		
		# start!
		if key == ord(' '):
			# countdown n sec
			video_show(oStream, "orange", "Recording starts in ", tuRectangle = (h, w), nCountdown = 3)
			
			# record video for n sec
			fElapsed, arFrames, _ = video_capture(oStream, "red", "Recording ", \
				tuRectangle = (h, w), nTimeDuration = int(diVideoSet["fDurationAvg"]), bOpticalFlow = False)
			print("\nCaptured video: %.1f sec, %s, %.1f fps" % \
				(fElapsed, str(arFrames.shape), len(arFrames)/fElapsed))

			# show orange wait box
			frame_show(oStream, "orange", "Translating sign ...", tuRectangle = (h, w))

			# predict video
			nLabel, sLabel, fProba = predict_frames(arFrames, keFeature, keLSTM, oClasses)
			sResults = "Identified sign: [%d] %s (confidence %.1f%%)" % (nLabel, sLabel, fProba*100.)
			print(sResults)
			nCount += 1

			# ready for next video	

		# debug
		elif key == ord('f'):
			sVideoFile = random.choice(liVideosDebug)
			arFrames = video2frames(sVideoFile, 256)
			print("DEBUG: Loaded %s | shape %s" % (sVideoFile, str(arFrames.shape)))
			frames_show(arFrames, int(video_length(sVideoFile)*1000 / len(arFrames)))
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
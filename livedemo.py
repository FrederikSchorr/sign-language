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
from model_mobile import features_2D_load_model
from model_lstm import lstm_load
from model_i3d import I3D_load
from predict import probability2label
from predict_mobile_lstm import predict_mobile_lstm


def livedemo():
	
	# dataset
	diVideoSet = {"sName" : "chalearn",
		"nClasses" : 20,   # number of classes
		"nFramesNorm" : 40,    # number of frames per video
		"nMinDim" : 240,   # smaller dimension of saved video-frames
		"tuShape" : (240, 320), # height, width
		"nFpsAvg" : 10,
		"nFramesAvg" : 50, 
		"fDurationAvg" : 5.0} # seconds 

	sAlgorithm = "i3d"  # "mobile-lstm" or "i3d"

	# files
	sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
	sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
	

	print("\nStarting gesture recognition live demo ... ")
	print(os.getcwd())
	print(diVideoSet)
	
	# load label description
	oClasses = VideoClasses(sClassFile)

	if sAlgorithm == "mobile-lstm":
		#sModelFile = "model/20180623-0429-04-chalearn010-otvl1-mobile-lstm-best.h5"
		#sModelFile = "model/20180626-1149-04-chalearn010-otvl1-mobile-lstm-best.h5"
		#sModelFile = "model/20180626-1458-chalearn010-flow-mobile-lstm-best.h5"
		sModelFile = "model/20180626-1939-chalearn020-flow-mobile-lstm-best.h5"

		# feature extractor 
		diFeature = {"sName" : "mobilenet",
			"tuInputShape" : (224, 224, 3),
			"tuOutputShape" : (1024, )}

		# Load pretrained MobileNet model without top layer 
		keMobile = features_2D_load_model(diFeature)
		h, w, c = diFeature["tuInputShape"]
		bThirdChannel = True

		# Load trained LSTM network
		keLSTM = lstm_load(sModelFile, diVideoSet["nFramesNorm"], diFeature["tuOutputShape"][0], oClasses.nClasses)

	elif sAlgorithm == "i3d":
		sModelFile = "model/20180627-0729-chalearn020-oflow-i3d-entire-best.h5"
		h, w, c = 224, 224, 2
		bThirdChannel = False
		keI3D = I3D_load(sModelFile, diVideoSet["nFramesNorm"], (h, w, c), oClasses.nClasses)

	# open a pointer to the webcam video stream
	oStream = video_start(device = 1, tuResolution = (320, 240), nFramePerSecond = diVideoSet["nFpsAvg"])

	#liVideosDebug = glob.glob(sVideoDir + "/train/*/*.*")
	nCount = 0
	sResults = ""
	timer = Timer()

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

			# crop and downsample frames
			arFrames = images_crop(arFrames, h, w)
			arFrames = frames_downsample(arFrames, diVideoSet["nFramesNorm"])

			# Translate frames to flows - these are already scaled between [-1.0, 1.0]
			print("Calculate optical flow on %d frames ..." % len(arFrames))
			timer.start()
			arFlows = frames2flows(arFrames, bThirdChannel = bThirdChannel, bShow = True)
			print("Optical flow per frame: %.3f" % (timer.stop() / len(arFrames)))

			# predict video from flows
			if sAlgorithm == "mobile+lstm": arProbas = predict_mobile_lstm(arFlows, keMobile, keLSTM)
			elif sAlgorithm == "i3d":
				print("Predict video with %s ..." % (keI3D.name))
				arX = np.expand_dims(arFlows, axis=0)
				arProbas = keI3D.predict(arX, verbose = 1)[0]

			# get/print top predictions+labels
			nLabel, sLabel, fProba = probability2label(arProbas, oClasses, nTop = 3)

			sResults = "Identified sign: [%d] %s (confidence %.1f%%)" % (nLabel, sLabel, fProba*100.)
			print(sResults)
			nCount += 1

		# quit
		elif key == ord('q'):
			break

		"""# debug
		elif key == ord('f'):
			sVideoFile = random.choice(liVideosDebug)
			arFrames = video2frames(sVideoFile, 256)
			print("DEBUG: Loaded %s | shape %s" % (sVideoFile, str(arFrames.shape)))
			frames_show(arFrames, int(video_length(sVideoFile)*1000 / len(arFrames)))
			nLabel, sLabel, fProba = predict_frames(arFrames, keFeature, keLSTM, oClasses)
			sResults = "Identified sign: [%d] %s (confidence %.0f%%)" % (nLabel, sLabel, fProba*100.)
		"""

	# do a bit of cleanup
	oStream.release()
	cv2.destroyAllWindows()

	return


if __name__ == '__main__':
	livedemo()
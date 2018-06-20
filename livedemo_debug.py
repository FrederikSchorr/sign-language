# import the necessary packages
import time
import os
import glob
import sys
import random

import numpy as np
import pandas as pd

import cv2


from frame import video2frames, images_normalize
from videocapture import frame_show, video_show, video_capture
from opticalflow import frames2flows
from datagenerator import VideoClasses
from feature_2D import features_2D_load_model
from train_mobile_lstm import lstm_load
from predict_mobile_lstm import predict


def list_videos(sVideoDir, nSamplesMax = None):
	""" retrieve all video files. Expected folder structure .../label/videofile.avi """
	print("Retrieve videos from %s ..." % sVideoDir)
	liVideos = glob.glob(os.path.join(sVideoDir, "*", "*.*"))
	nSamples = len(liVideos)
	
	if nSamples == 0: raise ValueError("Found no video files in %s" % sVideoDir)
	
	if nSamplesMax != None and nSamples > nSamplesMax:
		liVideos = random.sample(liVideos, nSamplesMax)
	
	print("Selected %d (out of %d) videos" % (len(liVideos), nSamples))
	return liVideos


def main():
	
	# important constants
	nClasses = 21   # number of classes
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
	sVideoDir	= "data-set/01-ledasila/%03d/train"%(nClasses)

	print("\nDebugging ... ")
	print(os.getcwd())
	
	# load label description
	oClasses = VideoClasses(sClassFile)

	# Load pretrained MobileNet model without top layer 
	keFeature = features_2D_load_model(diFeature)
	h, w, c = diFeature["tuInputShape"]

	# Load trained LSTM network
	keLSTM = lstm_load(sModelFile, nFramesNorm, diFeature["tuOutputShape"][0], oClasses.nClasses)

	# read video files into list
	liVideos = list_videos(sVideoDir, nSamplesMax = 100)

	nCount = 0
	nSuccess = 0
	# loop through all videos
	for sVideoPath in liVideos:
		
		# read videofile into frames
		print("\n%3d predict video %s ..." % (nCount, sVideoPath))
		arFrames = video2frames(sVideoPath, nMinDim)
		print("Frames: %s" % (str(arFrames.shape)))

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

		# compare groundtruth with predicted label
		sLabelTrue = sVideoPath.split("/")[-2]
		print("Groundtruth label: %s" % (sLabelTrue))
				
		if sLabel == sLabelTrue:
			nSuccess += 1
			print("CORRECT!                 ", end="")
		else:
			print("Not correctly predicted. ", end="")
		
		nCount += 1
		print("Accuracy after %3d samples: %.2f%%" % (nCount, nSuccess/nCount*100.0)) 

	return


if __name__ == '__main__':
	main()
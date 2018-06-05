# import the necessary packages
import time
import os
import glob
import sys
import random

import numpy as np
import pandas as pd

import cv2

from demo import load_inception, load_lstm, frames_to_features, classify_video

sys.path.append(os.path.abspath('03a-chalearn'))
from chalearn import Features


def list_videos(sVideoDir, nSamplesMax = None):
	""" retrieve all video files. Expected folder structure .../label/videofile.avi """
	print("Retrieve videos from %s ..." % sVideoDir)
	liVideos = glob.glob(os.path.join(sVideoDir, "*", "*.avi"))
	nSamples = len(liVideos)
	
	if nSamples == 0:
		print("ERROR: found no video files in %s" % sVideoDir)
		assert False
	
	if nSamplesMax != None and nSamples > nSamplesMax:
		liVideos = random.sample(liVideos, nSamplesMax)
	
	print("Selected %d (out of %d) videos" % (len(liVideos), nSamples))
	return liVideos


def video_to_frames(sVideoPath):
	""" read video file with OpenCV """

	# Create a VideoCapture object and read from input file
	print("Slice video %s to frames ..." % sVideoPath)
	oVideo = cv2.VideoCapture(sVideoPath)
	
	# Check if file opened successfully
	if (oVideo.isOpened()== False): 
		print("ERROR opening video file")
	
	liFrames = []
	# Read until video is completed
	while(oVideo.isOpened()):
		# Capture frame-by-frame
		(bGrabbed, arFrame) = oVideo.read()
		if bGrabbed == True:
			# Save the resulting frame
			liFrames.append(arFrame)			
		# finish
		else: 
			break 
	
	# When everything done, release the video object
	oVideo.release()

	return liFrames


def main():
	sModelFile = "03a-chalearn/model/20180525-1033-lstm-35878in249-best.h5"
	sLabelsFile = "datasets/04-chalearn/labels.csv"
	sVideoDir = "datasets/04-chalearn/train"
	sFeatureDir = "03a-chalearn/data/feature/val"

	nFramesNorm = 20
	nFeatureLength = 2048

	print("\nStarting test from " + os.getcwd())

	# load label description: nIndex,sLabel,sNameLong,sCat,sDetail
	dfLabels = pd.read_csv(sLabelsFile, header = 0, index_col = 0, dtype = {"sLabel":str})

	# read video files into list
	#liVideos = list_videos(sVideoDir, nSamplesMax = 10)
	# read features from disc
	oFeatures = Features(sFeatureDir, nFramesNorm, nFeatureLength)
	
	# load neural networks
	#nnCNN = load_inception()
	nnLSTM = load_lstm(sModelFile, nFramesNorm, nFeatureLength)

	nCount = 0
	nSuccess = 0
	# loop through all videos
	for i in range(oFeatures.nSamples):
		print("\nPredict video {}: {} ...".format(nCount, oFeatures.liPaths[i]))

		# slice video into frames
		#liFrames = video_to_frames(sVideoPath)
	
		#print("Read video with {} frames".format(len(liFrames)))
		#if len(liFrames) < nFramesNorm: continue

		# run CNN to translate video to feature
		#arFeatures = frames_to_features(nnCNN, liFrames, nFramesNorm)
		
		# run LSTM to predict label from features
		arLabel, arProba = classify_video(nnLSTM, oFeatures.arFeatures[i], nFramesNorm, nFeatureLength)

		# compare groundtruth with predicted label
		#sLabelTrue = sVideoPath.split("/")[-2]
		sLabelTrue = oFeatures.liLabels[i]
		sLabelTrueDetail = dfLabels.set_index("sLabel").loc[sLabelTrue, "sDetail"]
		print("Groundtruth label: [{}] {}".format(sLabelTrue, sLabelTrueDetail))
		
		sLabelPred = dfLabels.sLabel[arLabel[0]]
		print("Identified sign:   [{}] {} (confidence {:.0f}%)". \
			format(sLabelPred, dfLabels.sDetail[arLabel[0]], arProba[0]*100))
		
		if sLabelTrue == sLabelPred:
			nSuccess += 1
			print("CORRECT!                 ", end="")
		else:
			print("Not correctly predicted. ", end="")
		
		nCount += 1
		print("Accuracy after {} samples: {:.1f}".format(nCount, nSuccess/nCount*100.0)) 
	return


if __name__ == '__main__':
	main()
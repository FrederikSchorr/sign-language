# import the necessary packages
import time
import os
import sys

import numpy as np
import pandas as pd

import cv2
from videocapture import frame_show, video_show, video_capture

sys.path.append(os.path.abspath("03a-chalearn"))
from deeplearning import ConvNet, RecurrentNet, VideoFeatures


def frames_to_features(oCNN, liFrames, nFramesNorm):
	""" Extract CNN features from frames

	# Returns 
		An array with extracted features
	"""
	
	# check number of frames
	if len(liFrames) > nFramesNorm:
		# downsample the list of frames
		print("Downsample number of frames from %d to %d" % (len(liFrames), nFramesNorm))
		fraction = len(liFrames) / nFramesNorm
		index = [int(fraction * i) for i in range(nFramesNorm)]
		liNorm = [liFrames[i] for i in index]
		liFrames = liNorm

	elif (len(liFrames) < nFramesNorm):
		raise ValueError("Not enough frames, expected %d" % nFramesNorm)
	
	# Get the prediction
	print("Calculate CNN features from %d frames" % len(liFrames))
	arFrames = oCNN.resize_transform(liFrames)
	arFrames = oCNN.preprocess_input(arFrames)
	arFeatures = oCNN.keModel.predict(arFrames, verbose=1)
     
	return arFeatures


def classify_video(oRNN, arFeatures):
	""" Run the features (of the frames) through LSTM to classify video

    # Return
	 	(5 most probable labels, their probabilities) 
    """

	if arFeatures.shape != (oRNN.nFramesNorm, oRNN.nFeatureLength): raise ValueError("Incorrect shapes")

    # Only predict 1 sample
	arFeatures.resize(1, oRNN.nFramesNorm, oRNN.nFeatureLength) 

    # infer on RNN network
	print("Predict video category through {} ...".format(oRNN.sName))
	arPredictProba = oRNN.keModel.predict(arFeatures, verbose = 1)[0]

	arBestLabel = arPredictProba.argsort()[-5:][::-1]
	arBestProba = arPredictProba[arBestLabel]

	print("Most probable 5 classes:" + str(arBestLabel))
	print("with probabilites      :" + str(arBestProba))
	
	return arBestLabel, arBestProba


def main():
	sModelFile = "03a-chalearn/model/20180608-2306-lstm-35878in249-best.h5"
	sClassFile = "datasets/04-chalearn/class.csv"
	nFramesNorm = 20

	print("\nStarting Gesture recognition live demo from " + os.getcwd())
	
	# load label description
	dfClass = pd.read_csv(sClassFile)

	# load neural networks
	oCNN = ConvNet("mobilenet")
	oCNN.load_model()

	oRNN = RecurrentNet("lstm", nFramesNorm, oCNN.nOutputFeatures)
	oRNN.load_model(sModelFile)

	# open a pointer to the webcam video stream
	oStream = cv2.VideoCapture(0)

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
			fElapsed, liFrames = video_capture(oStream, "red", "Recording ", 5)
			print("\nCaptured video: {:.1f} sec, {} frames, {:.1f} fps". \
				format(fElapsed, len(liFrames), len(liFrames)/fElapsed))

			if len(liFrames) < nFramesNorm: 
				sResults = "No sufficiently long sign detected"
				continue

			# show orange wait box
			frame_show(oStream, "orange", "Translating sign ...")

			# run NN to translate video to label
			arFeatures = frames_to_features(oCNN, liFrames, nFramesNorm)
			arLabel, arProba = classify_video(oRNN, arFeatures)

			nPred = arLabel[0]
			sResults = "Identified sign: [{}] {} (confidence {:.0f}%)". \
				format(dfClass.sClass[nPred], dfClass.sDetail[nPred], arProba[0]*100)
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
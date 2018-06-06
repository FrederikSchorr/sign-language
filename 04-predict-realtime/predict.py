# import the necessary packages
import time
import os

import numpy as np
import pandas as pd

import cv2
from videocapture import frame_show, video_show, video_capture

from keras.applications import inception_v3
import keras.models


def load_inception():
	"""Load keras inception model with pretrained weights

	# Returns
		A keras Model instance
	"""
	print("Load pretrained Inception v3 model ...")
	nnBase = inception_v3.InceptionV3(weights='imagenet', include_top=True)
	
	# We'll extract features at the final pool layer
	nnCNN = keras.models.Model(
		inputs=nnBase.input,	
		outputs=nnBase.get_layer('avg_pool').output)
	
	nnCNN.summary()

	return nnCNN


def load_lstm(sModel, nFramesNorm, nFeatureLength):
	print("Load saved LSTM neural network %s ..." % sModel)
	nnLSTM = keras.models.load_model(sModel)
	nnLSTM.summary()

	assert(nnLSTM.input_shape == (None, nFramesNorm, nFeatureLength))
	return nnLSTM


def frames_to_features(nnCNN, liFrames, nFramesNorm):
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
		print("[ERROR] Not enough frames, expected %d" % nFramesNorm)
		return []

    # loop through frames to preprocess
	arX = np.zeros((nFramesNorm, 299, 299, 3))
	
	for i in range(nFramesNorm):
		frame = liFrames[i]
		frameResized = cv2.resize(frame, (299, 299))
		arX[i,:] = frameResized
	
	# Get the prediction
	print("Calculate CNN features from %d frames" % len(liFrames))
	arX = inception_v3.preprocess_input(arX)
	arFeatures = nnCNN.predict(arX, verbose=1)
     
	return arFeatures


def classify_video(nnLSTM, arFeatures, nFramesNorm, nFeatureLength):
	""" Run the features (of the frames) through LSTM to classify video

    # Return
	 	(5 most probable labels, their probabilities) 
    """

	assert(arFeatures.shape == (nFramesNorm, nFeatureLength))

    # Only predict 1 sample
	arFeatures.resize(1, nFramesNorm, nFeatureLength) 

    # infer on LSTM network
	print("Predict video category through LSTM ...")
	arPredictProba = nnLSTM.predict(arFeatures, verbose = 1)[0]

	arBestLabel = arPredictProba.argsort()[-5:][::-1]
	arBestProba = arPredictProba[arBestLabel]

	print("Most probable 5 classes:" + str(arBestLabel))
	print("with probabilites      :" + str(arBestProba))
	
	return arBestLabel, arBestProba


def main():
	sModelFile = "03a-chalearn/model/20180606-0802-lstm-35878in249-best.h5"
	sClassFile = "datasets/04-chalearn/class.csv"
	nFramesNorm = 20
	nFeatureLength = 2048

	print("\nStarting Gesture recognition live demo from " + os.getcwd())
	
	# load label description
	dfClass = pd.read_csv(sClassFile)

	# load neural networks
	nnCNN = load_inception()
	nnLSTM = load_lstm(sModelFile, nFramesNorm, nFeatureLength)

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
			arFeatures = frames_to_features(nnCNN, liFrames, nFramesNorm)
			arLabel, arProba = classify_video(nnLSTM, arFeatures, nFramesNorm, nFeatureLength)

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
"""
Classify human motion videos 
"""

import os
import glob
import sys
import warnings
import time

import numpy as np
import pandas as pd

import keras

from datagenerator import VideoClasses, FeaturesGenerator
from predict import predict_onfeature_generator


def predict_mobile_lstm(arFrames:np.array, keMobile, keLSTM) -> np.array:
	
	_, h, w, _ = keMobile.input_shape
	_, nFramesNorm, _ = keLSTM.input_shape

	# Calculate feature from flows
	print("Predict features with %s..." % keMobile.name)
	arFeatures = keMobile.predict(arFrames, batch_size = nFramesNorm // 4, verbose=1)

	# Classify flows with LSTM network
	print("Final predict with LSTM %s ..." % keLSTM.name)
	arX = np.expand_dims(arFeatures, axis=0)
	arProbas = keLSTM.predict(arX, verbose = 1)[0]

	return arProbas



def predict_onfeature_lstm():
   
    # important constants
    nClasses = 21   # number of classes
    nFrames = 20    # number of frames per video

    # feature extractor 
    diFeature = {"sName" : "mobilenet",
        "tuInputShape" : (224, 224, 3),
        "tuOutputShape" : (1024, )}
    #diFeature = {"sName" : "inception",
    #    "tuInputShape" : (299, 299, 3),
    #    "nOutput" : 2048}

    # directories
    sClassFile       = "data-set/01-ledasila/%03d/class.csv"%(nClasses)
    #sVideoDir        = "data-set/01-ledasila/%03d"%(nClasses)
    #sImageDir        = "data-temp/01-ledasila/%03d/image"%(nClasses)
    sImageFeatureDir = "data-temp/01-ledasila/%03d/image-mobilenet/val"%(nClasses)
    #sOflowDir        = "data-temp/01-ledasila/%03d/oflow"%(nClasses)
    sOflowFeatureDir = "data-temp/01-ledasila/%03d/oflow-mobilenet/val"%(nClasses)

    sModelDir        = "model"
    sModelImageBest  = sModelDir + "/20180620-1653-ledasila021-image-mobile-lstm-best.h5"
    sModelImageLast  = sModelDir + "/20180620-1653-ledasila021-image-mobile-lstm-last.h5"
    sModelOflowBest  = sModelDir + "/20180620-1701-ledasila021-oflow-mobile-lstm-best.h5"
    sModelOflowLast  = sModelDir + "/20180620-1701-ledasila021-oflow-mobile-lstm-last.h5"


    print("\nPredict with combined rgb + flow models ... ")
    print(os.getcwd())
    
    # initialize
    oClasses = VideoClasses(sClassFile)

    # predict on image features
    fAcc_image_best, _, arProba_image_best, liLabels_image_best = \
        predict_onfeature_generator(sImageFeatureDir, sModelImageBest, oClasses)
    print("Image best model accuracy: %.2f%%"%(fAcc_image_best*100.))

    fAcc_image_last, _, arProba_image_last, liLabels_image_last = \
        predict_onfeature_generator(sImageFeatureDir, sModelImageLast, oClasses)
    print("Image last model accuracy: %.2f%%"%(fAcc_image_last*100.))

    # predict on flow features
    fAcc_oflow_best, _, arProba_oflow_best, liLabels_oflow_best = \
        predict_onfeature_generator(sOflowFeatureDir, sModelOflowBest, oClasses)
    print("Optical flow best model accuracy: %.2f%%"%(fAcc_oflow_best*100.))

    fAcc_oflow_last, _, arProba_oflow_last, liLabels_oflow_last = \
        predict_onfeature_generator(sOflowFeatureDir, sModelOflowLast, oClasses)
    print("Optical flow last model accuracy: %.2f%%"%(fAcc_oflow_last*100.))

    # Combine predicted probabilities, first check if labels are equal
    if liLabels_image_best != liLabels_oflow_best:
        raise ValueError("The rgb and flwo labels do not match")

    # only last models
    arSoftmax = arProba_image_last + arProba_oflow_last # shape = (nSamples, nClasses)
    arProba = np.exp(arSoftmax) / np.sum(np.exp(arSoftmax), axis=1).reshape(-1,1)
    arPred = arProba.argmax(axis=1)
    fAcc = np.mean(liLabels_image_best == oClasses.dfClass.loc[arPred, "sClass"])
    print("\nLast image & last optical flow model combined accuracy: %.2f%%"%(fAcc*100.))

    # only flow models together
    arSoftmax = arProba_oflow_best + arProba_oflow_last # shape = (nSamples, nClasses)
    arProba = np.exp(arSoftmax) / np.sum(np.exp(arSoftmax), axis=1).reshape(-1,1)
    arPred = arProba.argmax(axis=1)
    fAcc = np.mean(liLabels_image_best == oClasses.dfClass.loc[arPred, "sClass"])
    print("\nCombined 2 optical flow model accuracy: %.2f%%"%(fAcc*100.))

    # average over all 4 models
    arSoftmax = arProba_image_best + arProba_image_last + arProba_oflow_best + arProba_oflow_last # shape = (nSamples, nClasses)
    arProba = np.exp(arSoftmax) / np.sum(np.exp(arSoftmax), axis=1).reshape(-1,1)
    arPred = arProba.argmax(axis=1)
    fAcc = np.mean(liLabels_image_best == oClasses.dfClass.loc[arPred, "sClass"])
    print("\nCombined 4-model accuracy: %.2f%%"%(fAcc*100.))

    return


if __name__ == '__main__':
    predict_onfeature_lstm()
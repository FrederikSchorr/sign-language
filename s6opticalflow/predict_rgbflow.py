"""
Classify human motion videos from ChaLearn dataset

ChaLearn dataset:
http://chalearnlap.cvc.uab.es/dataset/21/description/

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""

import numpy as np
import pandas as pd

import os
import glob
import sys

import warnings

import time

sys.path.append(os.path.abspath("."))
from s3chalearn.deeplearning import ConvNet, RecurrentNet, VideoClasses, VideoFeatures


def lstm_predict(sFeatureDir:str, sModelPath:str, nFramesNorm:int, nFeatureLength:int, oClasses:VideoClasses):    
    """ Predict labels for all features in given directory on saved model 
    
    Returns 
        fAccuracy
        arPredictions (dim = nSamples)
        arProbabilities (dim = (nSamples, nClasses)) 
        list of labels (groundtruth)
    """
    
    print("\nPredict features on LSTM ...")
    
    # Load video features
    oFeatures = VideoFeatures(sFeatureDir, nFramesNorm, nFeatureLength, oClasses.liClasses)
    if oFeatures.nSamples == 0: raise ValueError("LSTM prediction stopped")

    # load the LSTM network
    oRNN = RecurrentNet("lstm", nFramesNorm, nFeatureLength, oClasses)
    oRNN.load_model(sModelPath) 

    # predict
    arProba = oRNN.keModel.predict(
        oFeatures.arFeatures, 
        batch_size = None, 
        verbose = 1)

    arPred = arProba.argmax(axis=1)
    fAcc = np.mean(oFeatures.liLabels == oClasses.dfClass.loc[arPred, "sClass"])
    
    return fAcc, arPred, arProba, oFeatures.liLabels


def main():
   
    nClasses = 249
    nFramesNorm = 20 # number of frames per video for feature calculation

    # directories
    sClassFile      = "data-set/04-chalearn/class.csv"
    #sVideoDir       = "data-set/04-chalearn"
    #sFrameDir       = "data-temp/04-chalearn/%03d-frame-20"%(nClasses)
    #sFlowDir        = "data-temp/04-chalearn/%03d-oflow-20"%(nClasses)
    sFrameFeatureDir= "data-temp/04-chalearn/%03d-frame-20-mobilenet/val"%(nClasses)
    sFlowFeatureDir = "data-temp/04-chalearn/%03d-oflow-20-mobilenet/val"%(nClasses)

    sModelDir       = "model"
    #sModelRGB       = sModelDir + "/20180612-0847-lstm-41662in249-best.h5"
    sModelRGB       = sModelDir + "/20180608-2306-lstm-35878in249-best.h5"
    sModelFlow      = sModelDir + "/20180618-1210-chalearn249-oflow-lstm-best.h5"

    print("\nPredict with combined rgb + flow LSTM models. \nCurrent directory:", os.getcwd())
    
    # initialize
    oClasses = VideoClasses(sClassFile)
    oCNN = ConvNet("mobilenet")

    # predict on rgb features
    fAcc_rgb, _, arProba_rgb, liLabels_rgb = lstm_predict(sFrameFeatureDir, sModelRGB, 
        nFramesNorm, oCNN.nOutputFeatures, oClasses)
    print("RGB model accuracy: %.2f%%"%(fAcc_rgb*100.))

    # predict on flow features
    fAcc_flow, _, arProba_flow, liLabels_flow = lstm_predict(sFlowFeatureDir, sModelFlow, 
        nFramesNorm, oCNN.nOutputFeatures, oClasses)
    print("Flow model accuracy: %.2f%%"%(fAcc_flow*100.))

    # Combine predicted probabilities, first check if labels are equal
    if liLabels_rgb != liLabels_flow:
        raise ValueError("The labels of frame features are not equal flow features!")

    arSoftmax = arProba_rgb + arProba_flow # shape = (nSamples, nClasses)
    arProba = np.exp(arSoftmax) / np.sum(np.exp(arSoftmax), axis=1).reshape(-1,1)
    #print("arProba shape: %s. Sum of 3rd sample: %f (should be 1.0)"%(str(arProba.shape), np.sum(arProba[3,...])))

    arPred = arProba.argmax(axis=1)
    fAcc = np.mean(liLabels_rgb == oClasses.dfClass.loc[arPred, "sClass"])
    print("Combined RGB + flow model accuracy: %.2f%%"%(fAcc*100.))

    return


if __name__ == '__main__':
    main()
"""
Classify human motion videos from ChaLearn dataset

ChaLearn dataset:
http://chalearnlap.cvc.uab.es/dataset/21/description/

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""

import os
import glob
import sys
import warnings
import time

import numpy as np
import pandas as pd

import keras

sys.path.append(os.path.abspath("."))
from s7i3d.datagenerator import VideoClasses, FramesGenerator, FeaturesGenerator
from s7i3d.i3d_inception import Inception_Inflated3d_Top


def predict_i3d_top(sFeatureDir:str, sModelPath:str, nBatchSize:int, oClasses:VideoClasses):    
    """ Predict labels for all features in given directory on saved model 
    
    Returns 
        fAccuracy
        arPredictions (dim = nSamples)
        arProbabilities (dim = (nSamples, nClasses)) 
        list of labels (groundtruth)
    """
    
    print("\nPredict %s with I3D top model %s ..." %(sFeatureDir, sModelPath))
    
    # load the I3D top network
    keModel = keras.models.load_model(sModelPath)

    # Load video features
    genFeatures = FeaturesGenerator(sFeatureDir, nBatchSize,
        keModel.input_shape[1:], oClasses.liClasses, bShuffle = False)
    if genFeatures.nSamples == 0: raise ValueError("No feature files detected, prediction stopped")

    # predict
    print("Predict I3D top with generator ...")
    arProba = keModel.predict_generator(
        generator = genFeatures, 
        workers = 1,                 
        use_multiprocessing = False,
        verbose = 1)
    if arProba.shape[0] != genFeatures.nSamples: raise ValueError("Unexpected number of predictions")

    arPred = arProba.argmax(axis=1)
    liLabels = list(genFeatures.dfSamples.sLabel)
    fAcc = np.mean(liLabels == oClasses.dfClass.loc[arPred, "sClass"])
    
    return fAcc, arPred, arProba, liLabels


def main():
   
    nClasses = 3
    nBatchSize = 16

    # directories
    sClassFile       = "data-set/04-chalearn/class.csv"
    #sVideoDir       = "data-set/04-chalearn"
    #sFrameDir        = "data-temp/04-chalearn/%03d/frame"%(nClasses)
    sFrameFeatureDir = "data-temp/04-chalearn/%03d/frame-i3d/val"%(nClasses)
    #sFlowDir         = "data-temp/04-chalearn/%03d/oflow"%(nClasses)
    sFlowFeatureDir  = "data-temp/04-chalearn/%03d/oflow-i3d/val"%(nClasses)

    sModelDir       = "model"
    sModelRGB       = sModelDir + "/20180619-1524-chalearn003-rgb-i3dtop-last.h5"
    sModelFlow      = sModelDir + "/20180619-1550-chalearn003-oflow-i3dtop-last.h5"

    print("\nPredict with combined rgb + flow I3D top models. ", os.getcwd())
    
    # initialize
    oClasses = VideoClasses(sClassFile)

    # predict on rgb features
    fAcc_rgb, _, arProba_rgb, liLabels_rgb = predict_i3d_top(sFrameFeatureDir, sModelRGB, nBatchSize, oClasses)
    print("RGB model accuracy: %.2f%%"%(fAcc_rgb*100.))

    # predict on flow features
    fAcc_flow, _, arProba_flow, liLabels_flow = predict_i3d_top(sFlowFeatureDir, sModelFlow, nBatchSize, oClasses)
    print("Flow model accuracy: %.2f%%"%(fAcc_flow*100.))

    # Combine predicted probabilities, first check if labels are equal
    if liLabels_rgb != liLabels_flow:
        raise ValueError("The rgb and flwo labels do not match")

    arSoftmax = arProba_rgb + arProba_flow # shape = (nSamples, nClasses)
    arProba = np.exp(arSoftmax) / np.sum(np.exp(arSoftmax), axis=1).reshape(-1,1)
    #print("arProba shape: %s. Sum of 3rd sample: %f (should be 1.0)"%(str(arProba.shape), np.sum(arProba[3,...])))

    arPred = arProba.argmax(axis=1)
    fAcc = np.mean(liLabels_rgb == oClasses.dfClass.loc[arPred, "sClass"])
    print("\nCombined RGB + flow model accuracy: %.2f%%"%(fAcc*100.))

    return


if __name__ == '__main__':
    main()
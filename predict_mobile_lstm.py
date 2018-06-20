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


def predict(sFeatureDir:str, sModelPath:str, oClasses:VideoClasses, nBatchSize:int = 16):    
    """ Predict labels for all features in given directory on saved model 
    
    Returns 
        fAccuracy
        arPredictions (dim = nSamples)
        arProbabilities (dim = (nSamples, nClasses)) 
        list of labels (groundtruth)
    """
        
    # load the I3D top network
    print("Load model %s ..." % sModelPath)
    keModel = keras.models.load_model(sModelPath)

    # Load video features
    genFeatures = FeaturesGenerator(sFeatureDir, nBatchSize,
        keModel.input_shape[1:], oClasses.liClasses, bShuffle = False)
    if genFeatures.nSamples == 0: raise ValueError("No feature files detected, prediction stopped")

    # predict
    print("Predict with generator on %s ..." % sFeatureDir)
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
    sModelImage      = sModelDir + "/20180620-1653-ledasila021-image-mobile-lstm-last.h5"
    sModelOflow      = sModelDir + "/20180620-1701-ledasila021-oflow-mobile-lstm-last.h5"

    print("\nPredict with combined rgb + flow models ... ")
    print(os.getcwd())
    
    # initialize
    oClasses = VideoClasses(sClassFile)

    # predict on rgb features
    fAcc_rgb, _, arProba_rgb, liLabels_rgb = predict(sImageFeatureDir, sModelImage, oClasses)
    print("RGB model accuracy: %.2f%%"%(fAcc_rgb*100.))

    # predict on flow features
    fAcc_flow, _, arProba_flow, liLabels_flow = predict(sOflowFeatureDir, sModelOflow, oClasses)
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
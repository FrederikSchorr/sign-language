"""
https://github.com/FrederikSchorr/sign-language

Utilities for predicting a output label with a neural network 
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


def probability2label(arProbas:np.array, oClasses:VideoClasses, nTop:int = 3) -> (int, str, float):
    """ 
    # Return
        3-tuple: predicted nLabel, sLabel, fProbability
        in addition print nTop most probable labels
    """

    arTopLabels = arProbas.argsort()[-nTop:][::-1]
    arTopProbas = arProbas[arTopLabels]

    for i in range(nTop):
        sClass = oClasses.dfClass.sClass[arTopLabels[i]] + " " + oClasses.dfClass.sDetail[arTopLabels[i]]
        print("Top %d: [%3d] %s (confidence %.1f%%)" % \
            (i+1, arTopLabels[i], sClass, arTopProbas[i]*100.))
        
    #sClass = oClasses.dfClass.sClass[arTopLabels[0]] + " " + oClasses.dfClass.sDetail[arTopLabels[0]]
    return arTopLabels[0], oClasses.dfClass.sDetail[arTopLabels[0]], arTopProbas[0]


def predict_onfeature_generator(sFeatureDir:str, sModelPath:str, oClasses:VideoClasses, nBatchSize:int = 16):    
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
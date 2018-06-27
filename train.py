"""
Classify human gesture videos
"""

import os
import glob
import time
import sys
import warnings

import numpy as np
import pandas as pd

import keras

from datagenerator import VideoClasses, FeaturesGenerator


def train_feature_generator(sFeatureDir:str, sModelDir:str, sLogPath:str, keModel:keras.Model, oClasses: VideoClasses,
    nBatchSize:int=16, nEpoch:int=100, fLearn:float=1e-4):

    # Load training data
    genFeaturesVal   = FeaturesGenerator(sFeatureDir + "/val", nBatchSize,
        keModel.input_shape[1:], oClasses.liClasses)
    genFeaturesTrain = FeaturesGenerator(sFeatureDir + "/train", nBatchSize, 
        keModel.input_shape[1:], oClasses.liClasses)

    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger(sLogPath.split(".")[0] + "-acc.csv")

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    checkpointLast = keras.callbacks.ModelCheckpoint(
        filepath = sModelDir + "/" + (sLogPath.split("/")[-1]).split(".")[0] + "-last.h5",
        verbose = 0)
    checkpointBest = keras.callbacks.ModelCheckpoint(
        filepath = sModelDir + "/" + (sLogPath.split("/")[-1]).split(".")[0] + "-best.h5",
        verbose = 1, save_best_only = True)

    optimizer = keras.optimizers.Adam(lr = fLearn)
    keModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Fit!
    print("Fit with generator, learning rate %f ..." % fLearn)
    keModel.fit_generator(
        generator = genFeaturesTrain,
        validation_data = genFeaturesVal,
        epochs = nEpoch,
        workers = 1, #4,                 
        use_multiprocessing = False, #True,
        verbose = 1,
        callbacks=[csv_logger, checkpointLast, checkpointBest])    
    
    return
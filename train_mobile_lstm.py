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


def lstm_build(nFramesNorm:int, nFeatureLength:int, nClasses:int, fDropout:float = 0.5) -> keras.Model:

    # Build new LSTM model
    print("Build and compile LSTM model ...")
    keModel = keras.models.Sequential()
    keModel.add(keras.layers.LSTM(nFeatureLength * 1, return_sequences=True,
        input_shape=(nFramesNorm, nFeatureLength),
        dropout=fDropout))
    keModel.add(keras.layers.LSTM(nFeatureLength * 1, return_sequences=False, dropout=fDropout))
    keModel.add(keras.layers.Dense(nClasses, activation='softmax'))

    keModel.summary()

    return keModel


def lstm_load(sPath:str, nFramesNorm:int, nFeatureLength:int, nClasses:int) -> keras.Model:

    print("Load trained LSTM model from %s ..." % sPath)
    keModel = keras.models.load_model(sPath)
    
    tuInputShape = keModel.input_shape[1:]
    tuOutputShape = keModel.output_shape[1:]
    print("Loaded input shape %s, output shape %s" % (str(tuInputShape), str(tuOutputShape)))

    if tuInputShape != (nFramesNorm, nFeatureLength):
        raise ValueError("Unexpected LSTM input shape")
    if tuOutputShape != (nClasses, ):
        raise ValueError("Unexpected LSTM output shape")

    return keModel


def train_generator(sFeatureDir:str, sModelDir:str, sLogPath:str, keModel:keras.Model, oClasses: VideoClasses,
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
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath = sModelDir + "/" + (sLogPath.split("/")[-1]).split(".")[0] + "-best.h5",
        verbose = 1, save_best_only = True)

    optimizer = keras.optimizers.Adam(lr = fLearn)
    # Use same optimizer as in https://github.com/deepmind/kinetics-i3d
    #optimizer = keras.optimizers.SGD(lr = fLearn, momentum = 0.9, decay = 1e-7)
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
        callbacks=[csv_logger, checkpointer])    
    
    # save model
    sModelSaved = sModelDir + "/" + (sLogPath.split("/")[-1]).split(".")[0] + "-last.h5"
    keModel.save(sModelSaved)

    return


def train_mobile_lstm(diVideoSet, bImage = True, bOflow = True):

    # feature extractor 
    diFeature = {"sName" : "mobilenet",
        "tuInputShape" : (224, 224, 3),
        "tuOutputShape" : (1024, )}
    #diFeature = {"sName" : "inception",
    #    "tuInputShape" : (299, 299, 3),
    #    "nOutput" : 2048}

    # directories
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    #sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    #sImageDir        = "data-temp/%s/%03d/image"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sImageFeatureDir = "data-temp/%s/%03d/image-mobilenet"%(diVideoSet["sName"], diVideoSet["nClasses"])
    #sOflowDir        = "data-temp/%s/%03d/oflow"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sOflowFeatureDir = "data-temp/%s/%03d/oflow-mobilenet"%(diVideoSet["sName"], diVideoSet["nClasses"])

    sModelDir        = "model"

    print("\nStarting training with MobileNet + LSTM ...")
    print(os.getcwd())

    # read the classes
    oClasses = VideoClasses(sClassFile)

    # Image: Load LSTM and train it
    if bImage:
        sLogPath = "log/" + time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
            "-%s%03d-image-mobile-lstm.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
        print("Image log: %s" % sLogPath)

        keModelImage = lstm_build(diVideoSet["nFramesNorm"], diFeature["tuOutputShape"][0], oClasses.nClasses, fDropout = 0.5)
        train_generator(sImageFeatureDir, sModelDir, sLogPath, keModelImage, oClasses,
            nBatchSize = 16, nEpoch = 100, fLearn = 1e-4)

    # Oflow: Load LSTM and train it
    if bOflow:
        sLogPath = "log/" + time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
            "-%s%03d-otvl1-mobile-lstm.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
        print("Optical flow log: %s" % sLogPath)

        keModelOflow = lstm_build(diVideoSet["nFramesNorm"], diFeature["tuOutputShape"][0], oClasses.nClasses, fDropout = 0.5)
        train_generator(sOflowFeatureDir, sModelDir, sLogPath, keModelOflow, oClasses,
            nBatchSize = 16, nEpoch = 100, fLearn = 1e-4)

    return
    
    
if __name__ == '__main__':

    """diVideoSet = {"sName" : "01-ledasila",
        "nClasses" : 21,   # number of classes
        "nFramesNorm" : 64,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (288, 352), # height, width
        "nFpsAvg" : 25,
        "nFramesAvg" : 75,
        "fDurationAvg" : 3.0} # seconds
    """

    diVideoSet = {"sName" : "04-chalearn",
        "nClasses" : 10,   # number of classes
        "nFramesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (240, 320), # height, width
        "nFpsAvg" : 10,
        "nFramesAvg" : 50, 
        "fDurationAvG" : 5.0} # seconds 

    train_mobile_lstm(diVideoSet, bImage = False, bOflow = True)
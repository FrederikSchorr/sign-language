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
from train import train_feature_generator
from model_lstm import lstm_build


def train_mobile_lstm(diVideoSet, bImage = True, bOflow = True):

    # feature extractor 
    diFeature = {"sName" : "mobilenet",
        "tuInputShape" : (224, 224, 3),
        "tuOutputShape" : (1024, )}
    #diFeature = {"sName" : "inception",
    #    "tuInputShape" : (299, 299, 3),
    #    "nOutput" : 2048}

    # directories
    sFolder = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sImageDir        = "data-temp/%s/%s/image"%(diVideoSet["sName"], sFolder)
    sImageFeatureDir = "data-temp/%s/%s/image-mobilenet"%(diVideoSet["sName"], sFolder)
    sOflowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    sOflowFeatureDir = "data-temp/%s/%s/oflow-mobilenet"%(diVideoSet["sName"], sFolder)

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
        train_feature_generator(sImageFeatureDir, sModelDir, sLogPath, keModelImage, oClasses,
            nBatchSize = 16, nEpoch = 100, fLearn = 1e-4)

    # Oflow: Load LSTM and train it
    if bOflow:
        sLogPath = "log/" + time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
            "-%s%03d-flow-mobile-lstm.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
        print("Optical flow log: %s" % sLogPath)

        keModelOflow = lstm_build(diVideoSet["nFramesNorm"], diFeature["tuOutputShape"][0], oClasses.nClasses, fDropout = 0.5)
        train_feature_generator(sOflowFeatureDir, sModelDir, sLogPath, keModelOflow, oClasses,
            nBatchSize = 16, nEpoch = 100, fLearn = 1e-4)

    return
    
    
if __name__ == '__main__':

    """diVideoSet = {"sName" : "ledasila",
        "nClasses" : 21,   # number of classes
        "nFramesNorm" : 64,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (288, 352), # height, width
        "nFpsAvg" : 25,
        "nFramesAvg" : 75,
        "fDurationAvg" : 3.0} # seconds
    """

    diVideoSet = {"sName" : "chalearn",
        "nClasses" : 20,   # number of classes
        "nFramesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (240, 320), # height, width
        "nFpsAvg" : 10,
        "nFramesAvg" : 50, 
        "fDurationAvG" : 5.0} # seconds 

    train_mobile_lstm(diVideoSet, bImage = True, bOflow = True)
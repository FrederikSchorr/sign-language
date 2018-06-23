"""
From videos extract image frames and optical flow

Assume videos are stored: 
... sVideoDir / train / class001 / gesture.avi
... sVideoDir / val   / class249 / gesture.avi
"""

import os
import glob
import sys
import warnings

import numpy as np
import pandas as pd

from frame import videosDir2framesDir
from opticalflow import framesDir2flowsDir
from feature_2D import features_2D_load_model, features_2D_predict_generator


def prepare_mobile_lstm(diVideoSet):

    # feature extractor 
    diFeature = {"sName" : "mobilenet",
        "tuInputShape" : (224, 224, 3),
        "tuOutputShape" : (1024, )}
    #diFeature = {"sName" : "inception",
    #    "tuInputShape" : (299, 299, 3),
    #    "nOutput" : 2048}

    # directories
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sImageDir        = "data-temp/%s/%03d/image"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sImageFeatureDir = "data-temp/%s/%03d/image-mobilenet"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sOflowDir        = "data-temp/%s/%03d/otvl1"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sOflowFeatureDir = "data-temp/%s/%03d/otvl1-mobilenet"%(diVideoSet["sName"], diVideoSet["nClasses"])

    print("Extracting frames, optical flow and MobileNet features ...")
    print(os.getcwd())
    
    # extract frames from videos
    videosDir2framesDir(sVideoDir, sImageDir, diVideoSet["nMinDim"])

    # calculate optical flow
    framesDir2flowsDir(sImageDir, sOflowDir)

    # Load pretrained MobileNet model without top layer 
    keModel = features_2D_load_model(diFeature)

    # calculate MobileNet features from rgb frames
    features_2D_predict_generator(sImageDir + "/val",   sImageFeatureDir + "/val",   keModel, diVideoSet["nFramesNorm"])
    features_2D_predict_generator(sImageDir + "/train", sImageFeatureDir + "/train", keModel, diVideoSet["nFramesNorm"])

    # calculate MobileNet features from optical flow
    features_2D_predict_generator(sOflowDir + "/val",   sOflowFeatureDir + "/val",   keModel, diVideoSet["nFramesNorm"])
    features_2D_predict_generator(sOflowDir + "/train", sOflowFeatureDir + "/train", keModel, diVideoSet["nFramesNorm"])

    return


if __name__ == '__main__':
    
    diVideoSet = {"sName" : "04-chalearn",
        "nClasses" : 10,   # number of classes
        "nFramesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (240, 320), # height, width
        "nFpsAvg" : 10,
        "nFramesAvg" : 50, 
        "fDurationAvG" : 5.0} # seconds 
    #diVideoSet = {"sName" : "01-ledasila",
    #    "tuShape" : (288, 352), # height, width
    #    "nFPS" : 25,
    #    "nFrames" : 75,
    #    "fDuration" : 3.0} # seconds

    prepare_mobile_lstm(diVideoSet)
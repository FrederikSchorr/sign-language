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
from feature_2D import features_2D_load_model, features_2D_extract


def main():

    # important constants
    nClasses = 21   # number of classes
    nFrames = 20    # number of frames per video
    nMinDim = 288   # smaller dimension of saved video-frames. typically same as video 

    # feature extractor 
    diFeature = {"sName" : "mobilenet",
        "tuInputShape" : (224, 224, 3),
        "tuOutputShape" : (1024, )}
    #diFeature = {"sName" : "inception",
    #    "tuInputShape" : (299, 299, 3),
    #    "nOutput" : 2048}

    # directories
    sClassFile       = "data-set/01-ledasila/%03d/class.csv"%(nClasses)
    sVideoDir        = "data-set/01-ledasila/%03d"%(nClasses)
    sImageDir        = "data-temp/01-ledasila/%03d/image"%(nClasses)
    sImageFeatureDir = "data-temp/01-ledasila/%03d/image-mobilenet"%(nClasses)
    sOflowDir        = "data-temp/01-ledasila/%03d/oflow"%(nClasses)
    sOflowFeatureDir = "data-temp/01-ledasila/%03d/oflow-mobilenet"%(nClasses)

    print("Extracting frames, optical flow and MobileNet features ...")
    print(os.getcwd())
    
    # extract frames from videos
    videosDir2framesDir(sVideoDir, sImageDir, nMinDim)

    # calculate optical flow
    framesDir2flowsDir(sImageDir, sOflowDir)

    # Load pretrained MobileNet model without top layer 
    print("Load pretrained %s model for feature extraction ..." % (diFeature["sName"]))
    keModel = features_2D_load_model(diFeature)

    # calculate MobileNet features from rgb frames
    features_2D_extract(sImageDir + "/val",   sImageFeatureDir + "/val",   keModel, nFrames)
    features_2D_extract(sImageDir + "/train", sImageFeatureDir + "/train", keModel, nFrames)

    # calculate MobileNet features from optical flow
    features_2D_extract(sOflowDir + "/val",   sOflowFeatureDir + "/val",   keModel, nFrames)
    features_2D_extract(sOflowDir + "/train", sOflowFeatureDir + "/train", keModel, nFrames)

    return


if __name__ == '__main__':
    main()
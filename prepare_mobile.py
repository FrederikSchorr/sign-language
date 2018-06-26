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


def prepare_mobile(diVideoSet, bImage = True, bOflow = True):

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

    print("Extracting frames, optical flow and MobileNet features ...")
    print(os.getcwd())
    
    # extract frames from videos
    if bImage or bOflow:
        videosDir2framesDir(sVideoDir, sImageDir, nFramesNorm = diVideoSet["nFramesNorm"],
            nResizeMinDim = diVideoSet["nMinDim"], tuCropShape = diFeature["tuInputShape"][0:2])

    # calculate optical flow
    if bOflow:
        framesDir2flowsDir(sImageDir, sOflowDir, nFramesNorm = diVideoSet["nFramesNorm"])

    # Load pretrained MobileNet model without top layer 
    if bImage or bOflow:
        keModel = features_2D_load_model(diFeature)

    # calculate MobileNet features from rgb frames
    if bImage:
        features_2D_predict_generator(sImageDir + "/val",   sImageFeatureDir + "/val",   keModel, diVideoSet["nFramesNorm"])
        features_2D_predict_generator(sImageDir + "/train", sImageFeatureDir + "/train", keModel, diVideoSet["nFramesNorm"])

    # calculate MobileNet features from optical flow
    if bOflow:
        features_2D_predict_generator(sOflowDir + "/val",   sOflowFeatureDir + "/val",   keModel, diVideoSet["nFramesNorm"])
        features_2D_predict_generator(sOflowDir + "/train", sOflowFeatureDir + "/train", keModel, diVideoSet["nFramesNorm"])

    return


if __name__ == '__main__':
    
    diVideoSet = {"sName" : "chalearn",
        "nClasses" : 20,   # number of classes
        "nFramesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (240, 320), # height, width
        "nFpsAvg" : 10,
        "nFramesAvg" : 50, 
        "fDurationAvg" : 5.0} # seconds 

    prepare_mobile(diVideoSet, bImage = False, bOflow = True)
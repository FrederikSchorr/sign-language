"""
From videos extract image frames and optical flow

Assume videos are stored: 
... sVideoDir / train / class001 / gesture.mp4
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
from datagenerator import VideoClasses
from model_i3d import Inception_Inflated3d, Inception_Inflated3d_Top
from feature import features_3D_predict_generator
from train_i3d import train_I3D

# Image and/or Optical flow pipeline
bImage = True
bOflow = True

# dataset
diVideoSet = {"sName" : "chalearn",
    "nClasses" : 20,   # number of classes
    "nFramesNorm" : 40,    # number of frames per video
    "nMinDim" : 240,   # smaller dimension of saved video-frames
    "tuShape" : (240, 320), # height, width
    "nFpsAvg" : 10,
    "nFramesAvg" : 50, 
    "fDurationAvg" : 5.0} # seconds 

# directories
sFolder = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
sImageDir        = "data-temp/%s/%s/image"%(diVideoSet["sName"], sFolder)
sImageFeatureDir = "data-temp/%s/%s/image-i3d"%(diVideoSet["sName"], sFolder)
sOflowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
sOflowFeatureDir = "data-temp/%s/%s/oflow-i3d"%(diVideoSet["sName"], sFolder)

print("Starting I3D pipeline ...")
print(os.getcwd())

# extract frames from videos
if bImage or bOflow:
    videosDir2framesDir(sVideoDir, sImageDir, nFramesNorm = diVideoSet["nFramesNorm"],
        nResizeMinDim = diVideoSet["nMinDim"], tuCropShape = (224, 224))

# calculate optical flow
if bOflow:
    framesDir2flowsDir(sImageDir, sOflowDir, nFramesNorm = diVideoSet["nFramesNorm"])

# initialize
oClasses = VideoClasses(sClassFile)

# calculate I3D features from rgb frames
if bImage:
    # Load pretrained i3d image model without top layer 
    print("Load pretrained I3D image model ...")
    keI3Dimage = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics',
        input_shape = (diVideoSet["nFramesNorm"], 224, 224, 3))
    # predict features
    features_3D_predict_generator(sImageDir + "/val",   sImageFeatureDir + "/val",   keI3Dimage, diVideoSet["nFramesNorm"])
    features_3D_predict_generator(sImageDir + "/train", sImageFeatureDir + "/train", keI3Dimage, diVideoSet["nFramesNorm"])

# calculate I3D features from optical flow
if bOflow:
    # Load pretrained i3d flow model without top layer 
    print("Load pretrained I3D optical flow model ...")
    keI3DOflow = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics',
       input_shape = (diVideoSet["nFramesNorm"], 224, 224, 2))
    # predict features
    features_3D_predict_generator(sOflowDir + "/val",   sOflowFeatureDir + "/val",   keI3DOflow, diVideoSet["nFramesNorm"])
    features_3D_predict_generator(sOflowDir + "/train", sOflowFeatureDir + "/train", keI3DOflow, diVideoSet["nFramesNorm"])

# train LSTM network(s)
train_I3D(diVideoSet, bImage, bOflow)

# the end
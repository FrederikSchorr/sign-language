"""
Classify human motion videos from ChaLearn dataset

ChaLearn dataset:
http://chalearnlap.cvc.uab.es/dataset/21/description/

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""

import os
import glob
import time
import sys

import numpy as np
import pandas as pd

import keras

from datagenerator import VideoClasses
from train_mobile_lstm import train_generator
from model_i3d import Inception_Inflated3d_Top


def train_I3D(diVideoSet, bImage = True, bOflow = True):
   
    # directories
    sFolder = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    sImageDir        = "data-temp/%s/%s/image"%(diVideoSet["sName"], sFolder)
    sImageFeatureDir = "data-temp/%s/%s/image-i3d"%(diVideoSet["sName"], sFolder)
    sOflowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    sOflowFeatureDir = "data-temp/%s/%s/oflow-i3d"%(diVideoSet["sName"], sFolder)
    
    sModelDir        = "model"

    fLearn = 1e-3
    nEpochs = 100
    nBatchSize = 16

    print("\nStarting I3D top training ...")
    print(os.getcwd())

    # read the classes
    oClasses = VideoClasses(sClassFile)

    # Image: Load I3D and train it
    if bImage:
        sLogPath = "log/" + time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
            "-%s%03d-image-i3dtop.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
        print("Load I3D image top model ...")

        keI3DtopImage = Inception_Inflated3d_Top((4, 1, 1, 1024), oClasses.nClasses, dropout_prob=0.5)
        keI3DtopImage.summary()
        train_generator(sImageFeatureDir, sModelDir, sLogPath, keI3DtopImage, oClasses,
            nBatchSize, nEpochs, fLearn)

    # Optical flow: Load I3D and train it
    if bOflow:
        sLogPath = "log/" + time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
            "-%s%03d-oflow-i3dtop.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
        print("Load I3D optical flow top model ...")

        keI3DtopOflow = Inception_Inflated3d_Top((4, 1, 1, 1024), oClasses.nClasses, dropout_prob=0.5)
        #keI3DtopOflow.summary()
        train_generator(sOflowFeatureDir, sModelDir, sLogPath, keI3DtopOflow, oClasses,
            nBatchSize, nEpochs, fLearn)
      
    return
    
    
if __name__ == '__main__':

    diVideoSet = {"sName" : "chalearn",
        "nClasses" : 20,   # number of classes
        "nFramesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (240, 320), # height, width
        "nFpsAvg" : 10,
        "nFramesAvg" : 50, 
        "fDurationAvG" : 5.0} # seconds 

    train_I3D(diVideoSet, bImage = False, bOflow = True)
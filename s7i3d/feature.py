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
import warnings

import numpy as np
import pandas as pd

import keras

sys.path.append(os.path.abspath("."))
from s7i3d.preprocess import videosDir2framesDir
from s7i3d.datagenerator import VideoClasses, FramesGenerator
from s7i3d.i3d_inception import Inception_Inflated3d, add_top_layer


def framesDir2featuresDir(sFrameBaseDir:str, sFeatureBaseDir:str, keI3D:keras.Model, nBatchSize:int, oClasses:VideoClasses):

    # do not (partially) overwrite existing feature directory
    if os.path.exists(sFeatureBaseDir): 
        warnings.warn("\nFeature folder " + sFeatureBaseDir + " alredy exists, calculation stopped") 
        return

    # prepare frame generator
    _, nFrames, h, w, c = keI3D.input_shape
    genFrames = FramesGenerator(sFrameBaseDir, nBatchSize, nFrames, h, w, c, oClasses.liClasses)
    
    # Predict
    print("Predict I3D features with generator ...")
    arPredictions = keI3D.predict_generator(
        generator = genFrames,
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1)   
    print("I3D features shape: %s" % (str(arPredictions.shape)))    

    nPredictions = arPredictions.shape[0]
    if  nPredictions > genFrames.nSamples: raise ValueError("Unexpected output shape")

    # write features to files
    for i in range(nPredictions):
        # ... sFrameBaseDir / class / videoname=frame-directory
        sFrameDir = genFrames.dfVideos.sFrameDir[i]
        sVideoName = sFrameDir.split("/")[-1]
        sLabel = genFrames.dfVideos.sLabel[i]

        # save to file
        arFeature = arPredictions[i, ...]
        os.makedirs(sFeatureBaseDir + "/" + sLabel, exist_ok = True)
        np.save(sFeatureBaseDir + "/" + sLabel + "/" + sVideoName + ".npy", arFeature)

    print("%d I3D features saved to files in %s" % (i, sFeatureBaseDir))
    return


def main():
   
    nClasses = 3

    # directories
    sClassFile       = "data-set/04-chalearn/class.csv"
    #sVideoDir       = "data-set/04-chalearn"
    sFrameDir        = "data-temp/04-chalearn/%03d/frame"%(nClasses)
    sFrameFeatureDir = "data-temp/04-chalearn/%03d/frame-i3d"%(nClasses)
    sFlowDir         = "data-temp/04-chalearn/%03d/oflow"%(nClasses)
    sFlowFeatureDir  = "data-temp/04-chalearn/%03d/oflow-i3d"%(nClasses)

    NUM_FRAMES = 79
    FRAME_HEIGHT = 224
    FRAME_WIDTH = 224
    NUM_RGB_CHANNELS = 3
    NUM_FLOW_CHANNELS = 2

    BATCHSIZE = 4

    print("\nStarting ChaLearn optical flow to I3D features calculation in directory:", os.getcwd())

    # extract images
    #videosDir2framesDir(sVideoDir, sFrameDir, nClasses)

    # initialize
    oClasses = VideoClasses(sClassFile)

    # Load pretrained i3d flow model without top layer 
    print("Load pretrained I3D flow model ...")
    keI3D_flow = Inception_Inflated3d(
        include_top=False,
        weights='flow_imagenet_and_kinetics',
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_FLOW_CHANNELS))
    #keI3D_flow.summary() 

    # calculate features from optical flow
    framesDir2featuresDir(sFlowDir + "/val", sFlowFeatureDir + "/val", keI3D_flow, BATCHSIZE, oClasses)
    #framesDir2featuresDir(sFrameDir + "/train", sFeatureDir + "/train")

    return
    
    
if __name__ == '__main__':
    main()
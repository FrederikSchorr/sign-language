"""
https://github.com/FrederikSchorr/sign-language

In some video classification NN architectures it may be necessary to calculate features 
from the (video) frames, that are afterwards used for NN training.

Eg in the MobileNet-LSTM architecture, the video frames are first fed into the MobileNet
and the resulting 1024 **features** saved to disc.
"""

import os
import glob
import time
import sys
import warnings

import numpy as np
import pandas as pd

import keras

from datagenerator import FramesGenerator


def features_2D_predict_generator(sFrameBaseDir:str, sFeatureBaseDir:str, keModel:keras.Model, 
    nFramesNorm:int = 40):
    """
    Used by the MobileNet-LSTM NN architecture.
    The (video) frames (2-dimensional) in sFrameBaseDir are fed into keModel (eg MobileNet without top layers)
    and the resulting features are save to sFeatureBaseDir.
    """

    # do not (partially) overwrite existing feature directory
    #if os.path.exists(sFeatureBaseDir): 
    #    warnings.warn("\nFeature folder " + sFeatureBaseDir + " alredy exists, calculation stopped") 
    #    return

    # prepare frame generator - without shuffling!
    _, h, w, c = keModel.input_shape
    genFrames = FramesGenerator(sFrameBaseDir, 1, nFramesNorm, h, w, c, 
        liClassesFull = None, bShuffle=False)

    print("Predict features with %s ... " % keModel.name)
    nCount = 0
    # Predict - loop through all samples
    for _, seVideo in genFrames.dfVideos.iterrows():
        
        # ... sFrameBaseDir / class / videoname=frame-directory
        sVideoName = seVideo.sFrameDir.split("/")[-1]
        sLabel = seVideo.sLabel
        sFeaturePath = sFeatureBaseDir + "/" + sLabel + "/" + sVideoName + ".npy"

        # check if already predicted
        if os.path.exists(sFeaturePath):
            print("Video %5d: features already extracted to %s" % (nCount, sFeaturePath))
            nCount += 1
            continue

        # get normalized frames and predict feature
        arX, _ = genFrames.data_generation(seVideo)
        arFeature = keModel.predict(arX, verbose=0)

        # save to file
        os.makedirs(sFeatureBaseDir + "/" + sLabel, exist_ok = True)
        np.save(sFeaturePath, arFeature)

        print("Video %5d: features %s saved to %s" % (nCount, str(arFeature.shape), sFeaturePath))
        nCount += 1

    print("%d features saved to files in %s" % (nCount+1, sFeatureBaseDir))
    return
    

def features_3D_predict_generator(sFrameBaseDir:str, sFeatureBaseDir:str, 
    keModel:keras.Model, nBatchSize:int = 16):
    """
    Used by I3D-top-only model.
    The videos (frames) are fed into keModel (=I3D without top layers) and
    resulting features are saved to disc. 
    (Later these features are used to train a small model containing 
    only the adjusted I3D top layers.)
    """

    # do not (partially) overwrite existing feature directory
    #if os.path.exists(sFeatureBaseDir): 
    #    warnings.warn("\nFeature folder " + sFeatureBaseDir + " alredy exists, calculation stopped") 
    #    return

    # prepare frame generator - without shuffling!
    _, nFramesModel, h, w, c = keModel.input_shape
    genFrames = FramesGenerator(sFrameBaseDir, nBatchSize, nFramesModel, h, w, c, 
        liClassesFull = None, bShuffle=False)

    # Predict
    print("Predict features with %s ... " % keModel.name)

    nCount = 0
    # loop through all samples
    for _, seVideo in genFrames.dfVideos.iterrows():

        # ... sFrameBaseDir / class / videoname=frame-directory
        sVideoName = seVideo.sFrameDir.split("/")[-1]
        sLabel = seVideo.sLabel
        sFeaturePath = sFeatureBaseDir + "/" + sLabel + "/" + sVideoName + ".npy"

        # check if already predicted
        if os.path.exists(sFeaturePath):
            print("Video %5d: features already extracted to %s" % (nCount, sFeaturePath))
            nCount += 1
            continue

        # get normalized frames
        arFrames, _ = genFrames.data_generation(seVideo)

        # predict single sample
        arFeature = keModel.predict(np.expand_dims(arFrames, axis=0))[0]

        # save to file
        os.makedirs(sFeatureBaseDir + "/" + sLabel, exist_ok = True)
        np.save(sFeaturePath, arFeature)

        print("Video %5d: features %s saved to %s" % (nCount, str(arFeature.shape), sFeaturePath))
        nCount += 1

    print("%d features saved to files in %s" % (nCount+1, sFeatureBaseDir))
    
    return
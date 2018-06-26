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

from datagenerator import FramesGenerator


def features_2D_load_model(diFeature:dict) -> keras.Model:
    sModelName = diFeature["sName"]
    print("Load 2D extraction model %s ..." % sModelName)

    # load pretrained keras models
    if sModelName == "mobilenet":

        # load base model with top
        keBaseModel = keras.applications.mobilenet.MobileNet(
            weights="imagenet",
            input_shape = (224, 224, 3),
            include_top = True)

        # We'll extract features at the final pool layer
        keModel = keras.models.Model(
            inputs=keBaseModel.input,
            outputs=keBaseModel.get_layer('global_average_pooling2d_1').output,
            name=sModelName + " without top layer") 

    elif sModelName == "inception":

        # load base model with top
        keBaseModel = keras.applications.inception_v3.InceptionV3(
            weights='imagenet',
            include_top=True)

        # We'll extract features at the final pool layer
        keModel = keras.models.Model(
            inputs=keBaseModel.input,
            outputs=keBaseModel.get_layer('avg_pool').output,
            name=sModelName + " without top layer") 
    
    else: raise ValueError("Unknown 2D feature extraction model")

    # check input & output dimensions
    tuInputShape = keModel.input_shape[1:]
    tuOutputShape = keModel.output_shape[1:]
    print("Expected input shape %s, output shape %s" % (str(tuInputShape), str(tuOutputShape)))

    if tuInputShape != diFeature["tuInputShape"]:
        raise ValueError("Unexpected input shape")
    if tuOutputShape != diFeature["tuOutputShape"]:
        raise ValueError("Unexpected output shape")

    return keModel



def features_2D_predict_generator(sFrameBaseDir:str, sFeatureBaseDir:str, keModel:keras.Model, 
    nFramesNorm:int = 40):

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
"""
Classify human motion videos from ChaLearn dataset

ChaLearn dataset:
http://chalearnlap.cvc.uab.es/dataset/21/description/

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""

import numpy as np
import pandas as pd

from math import ceil

import os
import glob
import sys

import warnings

import time

sys.path.append(os.path.abspath("."))
from s3chalearn.deeplearning import ConvNet, RecurrentNet, VideoClasses, VideoFeatures

import keras.preprocessing


def frames2features(sFrameDir:str, sFeatureDir:str, oCNN:ConvNet, nFramesNorm:int, nClasses:int=None):
    """ Use pretrained CNN to calculate features from video-frames """

    # do not (partially) overwrite existing feature directory
    if os.path.exists(sFeatureDir): 
        warnings.warn("\nFeature folder " + sFeatureDir + " alredy exists, calculation stopped") 
        return

    sCurrentDir = os.getcwd()
    # get list of directories with frames: ... / sFrameDir/train/class/videodir/frames.jpg
    os.chdir(sFrameDir)
    dfVideos = pd.DataFrame(glob.glob("*/*/*"), dtype=str, columns=["sFrameDir"])
    os.chdir(sCurrentDir)
    print("Found %d directories=videos with frames" % len(dfVideos))

    # eventually restrict to first nLabels
    if nClasses != None:
        dfVideos.loc[:,"sLabel"] = dfVideos.sFrameDir.apply(lambda s: s.split("/")[-2])
        liClasses = sorted(dfVideos.sLabel.unique())[:nClasses]
        dfVideos = dfVideos[dfVideos["sLabel"].isin(liClasses)]
        print("Using only %d directories from %d classes" % (len(dfVideos), nClasses))

    # loop over all videos-directories. 
    # Feed all frames into ConvNet, save results in file per video
    nCounter = 0
    for _, seVideo in dfVideos.iterrows():

        # save resulting list of features in file in eg "data/feature/train/label/video.npy"
        sFeaturePath = os.path.join(sFeatureDir, seVideo.sFrameDir + ".npy")
        if (os.path.exists(sFeaturePath)):
            # if feature already etracted, skip
            print("%5d | Features %s already exist" % (nCounter, sFeaturePath))
            continue
        os.makedirs(os.path.dirname(sFeaturePath), exist_ok=True)

        # retrieve frame files - in ascending order
        sFrames = sorted(glob.glob(os.path.join(sFrameDir, seVideo.sFrameDir, "*.jpg")))
        print("%5d | Extracting features from %d frames to %s" % (nCounter, len(sFrames), sFeaturePath))
        if len(sFrames) != nFramesNorm: raise ValueError("Number of frames unexpected %d vs %d"%(nFramesNorm, len(sFrames)))

        # assemble array with all images
        liFrames = []
        for sFrame in sFrames:
            
            # load frame
            pilFrame = keras.preprocessing.image.load_img(sFrame, target_size=oCNN.tuHeightWidth)
            arFrame = keras.preprocessing.image.img_to_array(pilFrame)
            liFrames.append(arFrame)  
        
        # predict features with ConvNet
        arFrames = oCNN.preprocess_input(np.array(liFrames))
        arFeatures = oCNN.keModel.predict(arFrames)
       
        # flatten the features
        arFeatures = arFeatures.reshape(nFramesNorm, -1)

        # save to file
        np.save(sFeaturePath, arFeatures)
        nCounter += 1
    return



def main():
   
    nClasses = 249
    nFramesNorm = 20 # number of frames per video for feature calculation

    # directories
    sClassFile      = "data-set/04-chalearn/class.csv"
    #sVideoDir      = "data-set/04-chalearn"
    sFrameDir       = "data-temp/04-chalearn/%03d-frame-%d"%(nClasses, nFramesNorm)
    sFeatureDir     = "data-temp/04-chalearn/%03d-frame-%d-mobilenet"%(nClasses, nFramesNorm)

    print("\nStarting ChaLearn feature extraction in directory:", os.getcwd())

    # unzip ChaLearn videos and sort them in folders=label
    #unzip_sort_videos(sVideoDir, sVideoDir + "/_zip/train.zip", sVideoDir + "/_zip/train.txt")
    #unzip_sort_videos(sVideoDir, sVideoDir + "/_zip/val.zip", sVideoDir + "/_zip/val.txt")

    # extract frames from videos
    #video2frames(sVideoDir, sFrameDir, nFramesNorm, nClasses)
    
    # calculate features from frames
    oCNN = ConvNet("mobilenet")
    oCNN.load_model()
    frames2features(sFrameDir, sFeatureDir, oCNN, nFramesNorm, nClasses)

if __name__ == '__main__':
    main()
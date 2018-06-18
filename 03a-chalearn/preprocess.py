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
import shutil

from subprocess import call, check_output
import time

from videounzip import unzip_sort_videos


def video2frames(sVideoDir, sFrameDir, nFramesNorm = 20, nClasses = None):
    """ Extract frames from videos """
    
    # do not (partially) overwrite existing frame directory
    if os.path.exists(sFrameDir): raise ValueError("Folder {} alredy exists".format(sFrameDir)) 

    # get videos. Assume sVideoDir / train / class / video.avi
    dfVideos = pd.DataFrame(glob.glob(sVideoDir + "/*/*/*.avi"), columns=["sVideoPath"])
    print("Located {} videos in {}, extracting {} frames each to {} ..."\
        .format(len(dfVideos), sVideoDir, nFramesNorm, sFrameDir))

    # eventually restrict to first nLabels
    if nClasses != None:
        dfVideos.loc[:,"sLabel"] = dfVideos.sVideoPath.apply(lambda s: s.split("/")[-2])
        liClasses = sorted(dfVideos.sLabel.unique())[:nClasses]
        dfVideos = dfVideos[dfVideos["sLabel"].isin(liClasses)]
        print("Using only {} videos from {} classes".format(len(dfVideos), nClasses))

    nCounter = 0
    # loop through all videos and extract frames
    for sVideoPath in dfVideos.sVideoPath:
        
        # source path: ... / sVideoDir / train / class / video.avi
        li_sVideoPath = sVideoPath.split("/")
        if len(li_sVideoPath) < 4: raise ValueError("Video path should have min 4 components: {}".format(str(li_sVideoPath)))
        sVideoName = li_sVideoPath[-1].split(".")[0]

        # create frame directory for each video
        sDir = sFrameDir + "/" + li_sVideoPath[-3] + "/" + li_sVideoPath[-2] + "/" + sVideoName
        os.makedirs(sDir, exist_ok=True)
        
        # determine length of video in sec and deduce frame rate
        fVideoSec = int(check_output(["mediainfo", '--Inform=Video;%Duration%', sVideoPath]))/1000.0
        #nFramesPerSec = int(ceil(nFramesNorm / fVideoSec))
        fFramesPerSec = nFramesNorm / fVideoSec

        # call ffmpeg to extract frames from videos
        sFrames = sDir + "/frame-%03d.jpg"
        call(["ffmpeg", "-loglevel", "error" ,"-y", "-i", sVideoPath, \
            "-r", str(fFramesPerSec), "-frames", str(nFramesNorm), sFrames])
            
        # check right number of frames
        nFrames = len(glob.glob(sDir + "/*.jpg"))
        print("%5d | %s | %2.3fsec | %.1ffps | %df" % \
            (nCounter, sDir, fVideoSec, fFramesPerSec, nFrames))
        if nFrames != nFramesNorm: raise ValueError("Incorrect number of frames extracted")
        nCounter += 1

    # check number of created frames
    nFramesTotal = len(glob.glob(sFrameDir + "/*/*/*/*.jpg"))
    print("%d frames extracted from %d videos" % (nFramesTotal, len(dfVideos)))

    return


def main():
   
    # directories
    sClassFile = "data-set/04-chalearn/class.csv"
    sVideoDir = "data-set/04-chalearn"
    sFrameDir = "data-temp/04-chalearn/frame-003-20"

    nFramesNorm = 20 # number of frames per video for feature calculation

    print("\nStarting ChaLearn extraction in directory:", os.getcwd())

    # unzip ChaLearn videos and sort them in folders=label
    #unzip_sort_videos(sVideoDir, sVideoDir + "/_zip/train.zip", sVideoDir + "/_zip/train.txt")
    #unzip_sort_videos(sVideoDir, sVideoDir + "/_zip/val.zip", sVideoDir + "/_zip/val.txt")

    # extract frames from videos
    video2frames(sVideoDir, sFrameDir, nFramesNorm, nClasses = 3)

if __name__ == '__main__':
    main()
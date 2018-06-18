"""
From ChaLearn videoframes extract optical flow

Assume ChaLearn videoframes are stored: 
... sFrameDir / train / class001 / videoname / frames.jpg

Output:
... FlowDir / train / class001 / videoname / opticalflow.flo
"""

import os
import glob
import sys

import warnings

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath("06-i3d"))
from video2frame import video2frames, file2frames, frames_show
from opticalflow import frames2flows, flows2file, file2flows, flows2colorimages


def framesDir2flowsDir(sFrameBaseDir:str, sFlowBaseDir:str):
    """ Extract frames from videos 
    
    Input videoframe structure:
    ... sFrameDir / train / class001 / videoname / frames.jpg

    Output:
    ... sFlowDir / train / class001 / videoname / flow.jpg
    """

    # do not (partially) overwrite existing directory
    if os.path.exists(sFlowBaseDir): 
        warnings.warn("Optical flow folder " + sFlowBaseDir + " alredy exists: flow calculation stopped")
        return

    # get list of directories with frames: ... / sFrameDir/train/class/videodir/frames.jpg
    sCurrentDir = os.getcwd()
    os.chdir(sFrameBaseDir)
    liVideos = glob.glob("*/*/*")
    os.chdir(sCurrentDir)
    print("Found %d directories=videos with frames" % len(liVideos))

    # loop over all videos-directories
    nCounter = 0
    for sFrameDir in liVideos:

        # generate target directory
        sFlowDir = sFlowBaseDir + "/" + sFrameDir

        # retrieve frame files - in ascending order
        arFrames = file2frames(sFrameBaseDir + "/" + sFrameDir)
        print("%5d | Calculating optical flow from %3d frames to %s" % (nCounter, arFrames.shape[0], sFlowDir))

        # calculate and save optical flow
        arFlows = frames2flows(arFrames)
        flows2file(arFlows, sFlowDir)

        nCounter += 1      

    return


def main():
    nClasses = 20

    sFrameDir = "data-temp/04-chalearn/%03d-frame-20"%(nClasses)
    sFlowDir =  "data-temp/04-chalearn/%03d-oflow-20"%(nClasses)

    print("\nChaLearn videos: calculating optical flow from frames ...")

    framesDir2flowsDir(sFrameDir, sFlowDir)

    return


if __name__ == '__main__':
    main()
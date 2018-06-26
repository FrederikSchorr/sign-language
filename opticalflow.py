"""
Convert frames to opticalflow
"""

import os
import glob
import sys
import time

import warnings

import numpy as np
import pandas as pd

import cv2

from frame import files2frames, video2frames, frames_show, image_crop, images_crop
from timer import Timer



class OpticalFlow:
    def __init__(self, sName:str = "tvl1", bThirdChannel:bool = False, fBound:float = 20.):
        self.sName = sName
        self.bThirdChannel = bThirdChannel
        self.fBound = fBound
        self.arPrev = np.zeros((1,1))

        if sName == "tvl1":
            self.oTVL1 = cv2.DualTVL1OpticalFlow_create(warps = 1)
                # Mo 25.6.2018: (theta = 0.1, nscales = 1, scaleStep = 0.3, warps = 4, epsilon = 0.02)
                # Very Fast (theta = 0.1, nscales = 1, scaleStep = 0.5, warps = 1, epsilon = 0.1)
                # Default: (tau=0.25, lambda=0.15, theta=0.3, nscales=5, warps=5, epsilon=0.01, 
                #innnerIterations=30, outerIterations=10, scaleStep=0.8, gamma=0.0, 
                #medianFiltering=5, useInitialFlow=False)
        elif sName == "farnback":
            pass
        else: raise ValueError("Unknown optical flow type")
        
        return


    def first(self, arImage:np.array) -> np.array:

        h, w, _ = arImage.shape

        # save first image in black&white
        self.arPrev = cv2.cvtColor(arImage, cv2.COLOR_BGR2GRAY)

        # first flow = zeros
        arFlow = np.zeros((h, w, 2), dtype = np.float32)

        if self.bThirdChannel:
            self.arZeros = np.zeros((h, w, 1), dtype = np.float32)
            arFlow = np.concatenate((arFlow, self.arZeros), axis=2) 

        return arFlow


    def next(self, arImage:np.array) -> np.array:

        # first?
        if self.arPrev.shape == (1,1): return self.first(arImage)

        # get image in black&white
        arCurrent = cv2.cvtColor(arImage, cv2.COLOR_BGR2GRAY)

        if self.sName == "tvl1":
            arFlow = self.oTVL1.calc(self.arPrev, arCurrent, None)
        elif self.sName == "farnback":
            arFlow = cv2.calcOpticalFlowFarneback(self.arPrev, arCurrent, flow=None, 
                pyr_scale=0.5, levels=1, winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        else: raise ValueError("Unknown optical flow type")

        # only 2 dims
        arFlow = arFlow[:, :, 0:2]

        # truncate to +/-15.0, then rescale to [-1.0, 1.0]
        arFlow[arFlow > self.fBound] = self.fBound 
        arFlow[arFlow < -self.fBound] = -self.fBound
        arFlow = arFlow / self.fBound

        if self.bThirdChannel:
            # add third empty channel
            arFlow = np.concatenate((arFlow, self.arZeros), axis=2) 

        self.arPrev = arCurrent

        return arFlow



def frames2flows(arFrames:np.array(int), bThirdChannel:bool = False, fBound:float = 20.) -> np.array(float):
    """ Calculates optical flow from frames

    Returns:
        array of flow-arrays, each with dim (h, w, 2), 
        with "flow"-values truncated to [-15.0, 15.0] and then scaled to [-1.0, 1.0]
        If bThirdChannel = True a third channel with zeros is added
    """

    # initialize optical flow calculation
    oOpticalFlow = OpticalFlow("tvl1", bThirdChannel, fBound)
    
    liFlows = []
    # loop through all frames
    for i in range(len(arFrames)):
        # calc dense optical flow
        arFlow = oOpticalFlow.next(arFrames[i, ...])
        liFlows.append(arFlow)

    return np.array(liFlows)



def flows2file(arFlows:np.array(float), sTargetDir:str):
    """ Save array of flows (2 channels with values in [-1.0, 1.0]) 
    to jpg files (with 3 channels 0-255 each) in sTargetDir
    """

    n, h, w, c = arFlows.shape

    os.makedirs(sTargetDir, exist_ok=True)

    arZeros = np.zeros((h, w, 1), dtype = np.float32)

    for i in range(n):

        # add third empty channel
        ar_f_Flow = np.concatenate((arFlows[i, ...], arZeros), axis=2)

        # rescale to 0-255  
        ar_n_Flow = np.round((ar_f_Flow + 1.0) * 127.5).astype(np.uint8)

        cv2.imwrite(sTargetDir + "/flow%03d.jpg"%(i), ar_n_Flow)

    return



def file2flows(sDir:str, b3channels:bool = False) -> np.array:
    """ Read flow files from directory
    Expects 3-channel jpg files
    Output
        Default: array with 2-channel flow, with floats between [-1.0, 1.0]
        If b3channels = True: including 3rd channel from jpeg (should result in zero values)
    """

    # important to sort flow files upfront
    liFiles = sorted(glob.glob(sDir + "/*.jpg"))

    #liFilesU = sorted(glob.glob(sDir + "/u/*.jpg"))
    #liFilesV = sorted(glob.glob(sDir + "/v/*.jpg"))

    if len(liFiles) == 0: raise ValueError("No optical flow files found in " + sDir)
    #if len(liFilesU) != len(liFilesV): raise ValueError("Flow files: same number expected in u and v")

    liFlows = []
    # loop through frames
    for i in range(len(liFiles)):

        ar_n_Flow = cv2.imread(liFiles[i])
        h, w, c = ar_n_Flow.shape

        # optical flow only 2-dim
        if not b3channels: 
            ar_n_Flow = ar_n_Flow[:,:,0:2]

        # rescale from 0-255 to [-15.0, 15.0]
        ar_f_Flow = ((ar_n_Flow / 127.5) - 1.).astype(np.float32)
        #print("Ori: %3dto%3d | Rescaled: %4.1fto%4.1f" % \
        #    (np.min(ar_n_Flow), np.max(ar_n_Flow), np.min(ar_f_Flow), np.max(ar_f_Flow)))

        #arU = cv2.imread(liFilesU[i], cv2.IMREAD_GRAYSCALE)
        #arV = cv2.imread(liFilesV[i], cv2.IMREAD_GRAYSCALE)
        #arFlow = np.concatenate((arU[:,:,np.newaxis], arV[:,:,np.newaxis]), axis=2)
        
        liFlows.append(ar_f_Flow)

    return np.array(liFlows)


def flow2colorimage(ar_f_Flow:np.array(float)) -> np.array(int):

    h, w, c = ar_f_Flow.shape
    if not isinstance(ar_f_Flow[0,0,0], np.float32): 
        warnings.warn("Need to convert flows to float32")
        ar_f_Flow = ar_f_Flow.astype(np.float32)

    ar_n_hsv = np.zeros((h, w, 3), dtype = np.uint8)
    ar_n_hsv[...,1] = 255

    # get colors
    mag, ang = cv2.cartToPolar(ar_f_Flow[..., 0], ar_f_Flow[..., 1])
    ar_n_hsv[...,0] = ang * 180 / np.pi / 2
    ar_n_hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    ar_n_bgr = cv2.cvtColor(ar_n_hsv, cv2.COLOR_HSV2BGR)
    return ar_n_bgr

def flows2colorimages(arFlows:np.array) -> np.array:
    n, _, _, _ = arFlows.shape
    liImages = []
    for i in range(n):
        arImage = flow2colorimage(arFlows[i, ...])
        liImages.append(arImage)
    return np.array(liImages)


def framesDir2flowsDir(sFrameBaseDir:str, sFlowBaseDir:str):
    """ Extract frames from videos 
    
    Input videoframe structure:
    ... sFrameDir / train / class001 / videoname / frames.jpg

    Output:
    ... sFlowDir / train / class001 / videoname / flow.jpg
    """

    # do not (partially) overwrite existing directory
    if os.path.exists(sFlowBaseDir): 
        warnings.warn("\nOptical flow folder " + sFlowBaseDir + " alredy exists: flow calculation stopped")
        return

    # get list of directories with frames: ... / sFrameDir/train/class/videodir/frames.jpg
    sCurrentDir = os.getcwd()
    os.chdir(sFrameBaseDir)
    liVideos = sorted(glob.glob("*/*/*"))
    os.chdir(sCurrentDir)
    print("Found %d directories=videos with frames" % len(liVideos))

    # loop over all videos-directories
    nCounter = 0
    for sFrameDir in liVideos:

        # generate target directory
        sFlowDir = sFlowBaseDir + "/" + sFrameDir

        # retrieve frame files - in ascending order
        arFrames = files2frames(sFrameBaseDir + "/" + sFrameDir)
        print("%5d | Calculating optical flow from %s frames to %s" % (nCounter, str(arFrames.shape), sFlowDir))

        # calculate and save optical flow
        arFlows = frames2flows(arFrames)
        flows2file(arFlows, sFlowDir)

        nCounter += 1      

    return



def unittest_fromfile():
    print("Unittest opticalflow functions ...")
    timer = Timer()

    # read test video and show it
    #arFrames = video2frames("data-set/04-chalearn/train/c049/M_00831.avi", 240)
    arFrames = video2frames("data-set/01-ledasila/021/train/Banane/Banane---k---vid-20079---lsid-22591.mp4", 240)
    print("Video frames %s" % str(arFrames.shape))
    frames_show(arFrames, 50)

    # calc flow and save to disc
    timer.start()
    arFlows = frames2flows(arFrames)
    print("Optical flow per frame: %.3f" % (timer.stop() / len(arFrames)))

    #flows2file(arFlows, "data-temp/unittest")

    # show color flows
    arFlowImages = flows2colorimages(arFlows)
    frames_show(arFlowImages, 50)  

    # read flows from directory and display
    #arFlows2 = file2flows("data-temp/unittest", b3channels=True)
    #print(arFlows2.shape, np.mean(arFlows2[:,:,:,2]))
    #frames_show(arFlows2, 50)

    return


if __name__ == '__main__':
    unittest_fromfile()
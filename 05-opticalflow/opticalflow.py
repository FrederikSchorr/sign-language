"""
Convert frames to opticalflow
"""

import os
import glob
import sys

import warnings

import numpy as np
import pandas as pd

import cv2

sys.path.append(os.path.abspath("06-i3d"))
from video2frame import video2frames, frames_show


def frames2flows(arFrames:np.array(int), fBound:float = 15.) -> np.array(int):
    """ Calculates optical flow from frames

    Returns:
        array of flow-arrays, each with dim (h, w, 2), 
        with "flow"-values truncated to [-15.0, 15.0] and then scaled to [-1.0, 1.0]
    """
    n, h, w, c = arFrames.shape

    # initialize with first frame
    arPrev = cv2.cvtColor(arFrames[0, ...], cv2.COLOR_BGR2GRAY)

    #TVL1 = cv2.DualTVL1OpticalFlow_create(warps=1)
    liFlows = []
    # loop through all frames
    for i in range(n):
        
        # get frame in black&white
        arFrame = cv2.cvtColor(arFrames[i, ...], cv2.COLOR_BGR2GRAY)
        # calc dense optical flow
        ar_f_Flow = cv2.calcOpticalFlowFarneback(arPrev, arFrame, flow=None, 
            pyr_scale=0.5, levels=1, winsize=15, iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        #ar_f_Flow = TVL1.calc(arPrev, arFrame, None)

        #print("flow %d: shape %s | type %s| min %f | max %f" % \
        #    (i, str(ar_fFlow.shape), str(type(ar_fFlow[0,0,0])), np.min(ar_fFlow), np.max(ar_fFlow)))

        # only 2 dims
        ar_f_Flow = ar_f_Flow[:, :, 0:2]

        # truncate to +/-15.0, then rescale to [-1.0, 1.0]
        ar_f_Flow[ar_f_Flow > fBound] = fBound 
        ar_f_Flow[ar_f_Flow < -fBound] = -fBound
        ar_f_Flow = ar_f_Flow / fBound

        # save in list and to file
        liFlows.append(ar_f_Flow)
        arPrev = arFrame

    return np.array(liFlows)


def flows2file(arFlows:np.array(float), sTargetDir:str):
    """ Save array of flows (2 channels with values in [-1.0, 1.0] 
    to jpg files (with 3 channels 0-255 each) in sTargetDir
    """

    n, h, w, c = arFlows.shape

    os.makedirs(sTargetDir, exist_ok=True)
    #os.makedirs(sTargetDir + "/u", exist_ok=True)
    #os.makedirs(sTargetDir + "/v", exist_ok=True)

    arZeros = np.zeros((h, w, 1), dtype = np.float32)

    for i in range(n):

        # add third empty channel
        ar_f_Flow = np.concatenate((arFlows[i, ...], arZeros), axis=2)

        # rescale to 0-255  
        ar_n_Flow = np.round((ar_f_Flow + 1.0) * 127.5).astype(np.uint8)

        # save x, y flows to r and g channels, since opencv reverses the colors
        #cv2.imwrite(sTargetDir + "/flow%03d.png"%(i), ar_n_Flow[:,:,::-1])
        cv2.imwrite(sTargetDir + "/flow%03d.jpg"%(i), ar_n_Flow)

        # save horizontal + vertical to (compressed) jpg files
        #cv2.imwrite(sTargetDir + "/u/frame%03d.jpg"%(i), arFlow[:, :, 0])
        #cv2.imwrite(sTargetDir + "/v/frame%03d.jpg"%(i), arFlow[:, :, 1])

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


def unittest():
    print("Unittest opticalflow functions ...")

    # read test video and show it
    arFrames = video2frames("data-set/04-chalearn/train/c049/M_00831.avi", 240)
    frames_show(arFrames, 50)

    # calc flow and save to disc
    arFlows = frames2flows(arFrames)
    flows2file(arFlows, "data-temp/unittest")

    # show color flows
    arFlowImages = flows2colorimages(arFlows)
    frames_show(arFlowImages, 50)  

    # read flows from directory and display
    arFlows2 = file2flows("data-temp/unittest", b3channels=True)
    print(arFlows2.shape, np.mean(arFlows2[:,:,:,2]))
    frames_show(arFlows2, 50)

    return

if __name__ == '__main__':
    unittest()
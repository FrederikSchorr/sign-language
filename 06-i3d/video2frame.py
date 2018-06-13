"""
Video <=> frame utilities
"""

import os
import glob

import numpy as np
import pandas as pd

import cv2
#from subprocess import call


def resize_aspectratio(arImage: np.array, nMinDim:int = 224) -> np.array:
    nHeigth, nWidth, _ = arImage.shape

    if nWidth >= nHeigth:
        # wider than high => map heigth to 224
        fRatio = nMinDim / nHeigth
    else: 
        fRatio = nMinDim / nWidth

    return cv2.resize(arImage, dsize = (0,0), fx = fRatio, fy = fRatio, interpolation=cv2.INTER_LINEAR)


def video2frame(sVideoPath:str) -> np.array:
    """ Read video file with OpenCV and return array of frames

    Frames are resized preserving aspect ratio 
    so that the smallest dimension is 224 pixels, 
    with bilinear interpolation
    """
    
    # Create a VideoCapture object and read from input file
    oVideo = cv2.VideoCapture(sVideoPath)
    if (oVideo.isOpened() == False): raise ValueError("Error opening video file")

    liFrames = []

    # Read until video is completed
    while(True):
        
        (bGrabbed, arFrame) = oVideo.read()
        if bGrabbed == False: break

        # resize image
        arFrameResized = resize_aspectratio(arFrame, 224)

		# Save the resulting frame to list
        liFrames.append(arFrameResized)
   
    return np.array(liFrames)


def frame2file(arFrames:np.array, sTargetDir:str):
    """ Write array of frames to jpg files
    Input: arFrames = (number of frames, height, width, depth)
    """

    for nFrame in range(arFrames.shape[0]):
        cv2.imwrite(sTargetDir + "/frame%04d.jpg" % nFrame, arFrames[nFrame, :, :, :])
    return


def file2frame(sPath:str) -> np.array:
    # important to sort image files upfront
    liFiles = sorted(glob.glob(sPath + "/*.jpg"))
    if len(liFiles) == 0: raise ValueError("No frames found in " + sPath)

    liFrames = []
    # loop through frames
    for sFramePath in liFiles:
        arFrame = cv2.imread(sFramePath)
        liFrames.append(arFrame)

    return np.array(liFrames)
    
    
def frames_trim(arFrames:np.array, nFramesTarget:int) -> np.array:
    """ Adjust number of frames (eg 123) to nFramesTarget (eg 79)
    works also if originally less frames then nFramesTarget
    """

    nSamples, nHeight, nWidth, nDepth = arFrames.shape
    if nSamples == nFramesTarget: return arFrames

    # down/upsample the list of frames
    fraction = nSamples / nFramesTarget
    index = [int(fraction * i) for i in range(nFramesTarget)]
    liTarget = [arFrames[i,:,:,:] for i in index]
    #print("Change number of frames from %d to %d" % (nSamples, nFramesTarget))
    #print(index)

    return np.array(liTarget)
    
    
def image_crop(arFrames:np.array, nHeightTarget, nWidthTarget) -> np.array:
    """ crop each frame in array to specified size, choose centered image
    """
    nSamples, nHeight, nWidth, nDepth = arFrames.shape

    if (nHeight < nHeightTarget) or (nWidth < nWidthTarget):
        raise ValueError("Image height/width too small to crop to target size")

    # calc left upper corner
    sX = int(nWidth/2 - nWidthTarget/2)
    sY = int(nHeight/2 - nHeightTarget/2)

    arFrames = arFrames[:, sY:sY+nHeightTarget, sX:sX+nWidthTarget, :]

    return arFrames


def frames_show(arFrames:np.array, nWaitMilliSec:int = 100):

    nFrames, nHeight, nWidth, nDepth = arFrames.shape
    
    for i in range(nFrames):
        cv2.imshow("Frame", arFrames[i,...])
        cv2.waitKey(nWaitMilliSec)

    return


def frame2flow(liFrames:list, sFlowDir_u:str, sFlowDir_v:str, nBound = 15):
       
    liFlows = []

    # initialize with first frame
    arPrev = liFrames[0]
    arPrev = cv2.cvtColor(arPrev, cv2.COLOR_BGR2GRAY)

    nCount = 0
    # loop through all frames
    for arFrame in liFrames:
        arFrame = cv2.cvtColor(arFrame, cv2.COLOR_BGR2GRAY)
        
        arFlow = cv2.calcOpticalFlowFarneback(arPrev,arFrame,
            flow=None, pyr_scale=0.5, levels=1, winsize=15, iterations=2,
            poly_n=5, poly_sigma=1.1, flags=0)

        arFlow = (arFlow + nBound) * (255.0 / (2*nBound))
        arFlow = np.round(arFlow).astype(int)
        arFlow[arFlow >= 255] = 255
        arFlow[arFlow <= 0] = 0
        
        # save in list and to file
        liFlows.append(arFlow)
        cv2.imwrite(sFlowDir_u + "/farneback{:04d}.jpg".format(nCount), arFlow[:, :, 0])
        cv2.imwrite(sFlowDir_v + "/farneback{:04d}.jpg".format(nCount), arFlow[:, :, 1])

        arPrev = arFrame
        nCount += 1

    return liFlows
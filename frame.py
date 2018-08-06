"""
https://github.com/FrederikSchorr/sign-language

Extract frames from a video (or many videos). 
Plus some frame=image manipulation utilities.
"""

import os
import glob
import warnings
import random
from subprocess import check_output

import numpy as np
import pandas as pd

import cv2


def image_resize_aspectratio(arImage: np.array, nMinDim:int = 256) -> np.array:
    nHeigth, nWidth, _ = arImage.shape

    if nWidth >= nHeigth:
        # wider than high => map heigth to 224
        fRatio = nMinDim / nHeigth
    else: 
        fRatio = nMinDim / nWidth

    if fRatio != 1.0:
        arImage = cv2.resize(arImage, dsize = (0,0), fx = fRatio, fy = fRatio, interpolation=cv2.INTER_LINEAR)

    return arImage


def images_resize_aspectratio(arImages: np.array, nMinDim:int = 256) -> np.array:
    nImages, _, _, _ = arImages.shape
    liImages = []
    for i in range(nImages):
        arImage = image_resize_aspectratio(arImages[i, ...])
        liImages.append(arImage)
    return np.array(liImages)
    

def video2frames(sVideoPath:str, nResizeMinDim:int) -> np.array:
    """ Read video file with OpenCV and return array of frames
    The frame rate depends on the video (and cannot be set)

    if nMinDim != None: Frames are resized preserving aspect ratio 
        so that the smallest dimension is eg 256 pixels, with bilinear interpolation
    """
    
    # Create a VideoCapture object and read from input file
    oVideo = cv2.VideoCapture(sVideoPath)
    if (oVideo.isOpened() == False): raise ValueError("Error opening video file")

    liFrames = []

    # Read until video is completed
    while(True):
        
        (bGrabbed, arFrame) = oVideo.read()
        if bGrabbed == False: break

        if nResizeMinDim != None:
            # resize image
            arFrameResized = image_resize_aspectratio(arFrame, nResizeMinDim)

		# Save the resulting frame to list
        liFrames.append(arFrameResized)
   
    return np.array(liFrames)


def frames2files(arFrames:np.array, sTargetDir:str):
    """ Write array of frames to jpg files
    Input: arFrames = (number of frames, height, width, depth)
    """
    os.makedirs(sTargetDir, exist_ok=True)
    for nFrame in range(arFrames.shape[0]):
        cv2.imwrite(sTargetDir + "/frame%04d.jpg" % nFrame, arFrames[nFrame, :, :, :])
    return


def files2frames(sPath:str) -> np.array:
    # important to sort image files upfront
    liFiles = sorted(glob.glob(sPath + "/*.jpg"))
    if len(liFiles) == 0: raise ValueError("No frames found in " + sPath)

    liFrames = []
    # loop through frames
    for sFramePath in liFiles:
        arFrame = cv2.imread(sFramePath)
        liFrames.append(arFrame)

    return np.array(liFrames)
    
    
def frames_downsample(arFrames:np.array, nFramesTarget:int) -> np.array:
    """ Adjust number of frames (eg 123) to nFramesTarget (eg 79)
    works also if originally less frames then nFramesTarget
    """

    nSamples, _, _, _ = arFrames.shape
    if nSamples == nFramesTarget: return arFrames

    # down/upsample the list of frames
    fraction = nSamples / nFramesTarget
    index = [int(fraction * i) for i in range(nFramesTarget)]
    liTarget = [arFrames[i,:,:,:] for i in index]
    #print("Change number of frames from %d to %d" % (nSamples, nFramesTarget))
    #print(index)

    return np.array(liTarget)
    
    
def image_crop(arFrame, nHeightTarget, nWidthTarget) -> np.array:
    """ crop 1 frame to specified size, choose centered image
    """
    nHeight, nWidth, _ = arFrame.shape

    if (nHeight < nHeightTarget) or (nWidth < nWidthTarget):
        raise ValueError("Image height/width too small to crop to target size")

    # calc left upper corner
    sX = int(nWidth/2 - nWidthTarget/2)
    sY = int(nHeight/2 - nHeightTarget/2)

    arFrame = arFrame[sY:sY+nHeightTarget, sX:sX+nWidthTarget, :]

    return arFrame


def images_crop(arFrames:np.array, nHeightTarget, nWidthTarget) -> np.array:
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


def images_rescale(arFrames:np.array) -> np.array(float):
    """ Rescale array of images (rgb 0-255) to [-1.0, 1.0]
    """

    ar_fFrames = arFrames / 127.5
    ar_fFrames -= 1.

    return ar_fFrames


def images_normalize(arFrames:np.array, nFrames:int, nHeight:int, nWidth:int, bRescale:bool = True) -> np.array(float):
    """ Several image normalizations/preprocessing: 
        - downsample number of frames
        - crop to centered image
        - rescale rgb 0-255 value to [-1.0, 1.0] - only if bRescale == True

    Returns array of floats
    """

    # normalize the number of frames (assuming typically downsampling)
    arFrames = frames_downsample(arFrames, nFrames)

    # crop to centered image
    arFrames = images_crop(arFrames, nHeight, nWidth)

    if bRescale:
        # normalize to [-1.0, 1.0]
        arFrames = images_rescale(arFrames)
    else:
        if np.max(np.abs(arFrames)) > 1.0: warnings.warn("Images not normalized")

    return arFrames


def frames_show(arFrames:np.array, nWaitMilliSec:int = 100):

    nFrames, nHeight, nWidth, nDepth = arFrames.shape
    
    for i in range(nFrames):
        cv2.imshow("Frame", arFrames[i, :, :, :])
        cv2.waitKey(nWaitMilliSec)

    return


def video_length(sVideoPath:str) -> float:
    return int(check_output(["mediainfo", '--Inform=Video;%Duration%', sVideoPath]))/1000.0


def videosDir2framesDir(sVideoDir:str, sFrameDir:str, nFramesNorm:int = None, 
    nResizeMinDim:int = None, tuCropShape:tuple = None, nClasses:int = None):
    """ Extract frames from videos 
    
    Input video structure:
    ... sVideoDir / train / class001 / videoname.avi

    Output:
    ... sFrameDir / train / class001 / videoname / frames.jpg
    """

    # do not (partially) overwrite existing frame directory
    #if os.path.exists(sFrameDir): 
    #    warnings.warn("Frame folder " + sFrameDir + " already exists, frame extraction stopped")
    #    return 

    # get videos. Assume sVideoDir / train / class / video.mp4
    dfVideos = pd.DataFrame(sorted(glob.glob(sVideoDir + "/*/*/*.*")), columns=["sVideoPath"])
    print("Located {} videos in {}, extracting to {} ..."\
        .format(len(dfVideos), sVideoDir, sFrameDir))
    if len(dfVideos) == 0: raise ValueError("No videos found")

    # eventually restrict to first nLabels
    if nClasses != None:
        dfVideos.loc[:,"sLabel"] = dfVideos.sVideoPath.apply(lambda s: s.split("/")[-2])
        liClasses = sorted(dfVideos.sLabel.unique())[:nClasses]
        dfVideos = dfVideos[dfVideos["sLabel"].isin(liClasses)]
        print("Using only {} videos from first {} classes".format(len(dfVideos), nClasses))

    nCounter = 0
    # loop through all videos and extract frames
    for sVideoPath in dfVideos.sVideoPath:

        # assemble target diretory (assumed directories see above)
        li_sVideoPath = sVideoPath.split("/")
        if len(li_sVideoPath) < 4: raise ValueError("Video path should have min 4 components: {}".format(str(li_sVideoPath)))
        sVideoName = li_sVideoPath[-1].split(".")[0]
        sTargetDir = sFrameDir + "/" + li_sVideoPath[-3] + "/" + li_sVideoPath[-2] + "/" + sVideoName
        
        # check if frames already extracted
        if nFramesNorm != None and os.path.exists(sTargetDir):
            nFrames = len(glob.glob(sTargetDir + "/*.*"))
            if nFrames == nFramesNorm: 
                print("Video %5d already extracted to %s" % (nCounter, sTargetDir))
                nCounter += 1
                continue
            else: 
                print("Video %5d: Directory with %d instead of %d frames detected" % (nCounter, nFrames, nFramesNorm))
        
        # create target directory
        os.makedirs(sTargetDir, exist_ok = True)

        # slice videos into frames with OpenCV
        arFrames = video2frames(sVideoPath, nResizeMinDim)

        # length and fps
        fVideoSec = video_length(sVideoPath)
        nFrames = len(arFrames)
        fFPS = nFrames / fVideoSec   

        # downsample
        if nFramesNorm != None: 
            arFrames = frames_downsample(arFrames, nFramesNorm)

        # crop images
        if tuCropShape != None:
            arFrames = images_crop(arFrames, *tuCropShape)
        
        # write frames to .jpg files
        frames2files(arFrames, sTargetDir)         

        print("Video %5d | %5.1f sec | %d frames | %4.1f fps | saved %s in %s" % (nCounter, fVideoSec, nFrames, fFPS, str(arFrames.shape), sTargetDir))
        nCounter += 1      

    return


def unittest(sVideoDir, nSamples = 100):
    print("\nAnalyze video durations and fps from %s ..." % (sVideoDir))
    print(os.getcwd())

    liVideos = glob.glob(sVideoDir + "/*/*.mp4") + glob.glob(sVideoDir + "/*/*.avi")
    
    if len(liVideos) == 0: raise ValueError("No videos detected")

    fVideoSec_sum, nFrames_sum = 0, 0
    for i in range(nSamples):
        sVideoPath = random.choice(liVideos)
        #print("Video %s" % sVideoPath)

        # read video
        arFrames = video2frames(sVideoPath, 256)
        nFrames = len(arFrames)

        # determine length of video in sec and deduce frame rate
        fVideoSec = video_length(sVideoPath)
        fFPS = nFrames / fVideoSec

        fVideoSec_sum += fVideoSec
        nFrames_sum += nFrames

        print("%2d: Shape %s, duration %.1f sec, fps %.1f" % (i, str(arFrames.shape), fVideoSec, fFPS))

    nCount = i+1
    print("%d samples: Average video duration %.1f sec, fps %.1f" % (nSamples, fVideoSec_sum / nCount, nFrames_sum / fVideoSec_sum))

    return

if __name__ == '__main__':

    unittest("data-set/01-ledasila/021/train", 100)
    unittest("data-set/04-chalearn/010/train", 100)
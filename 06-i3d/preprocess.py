"""
From ChaLearn videos extract image frames and optical flow

Assume ChaLearn videos are stored: 
... sVideoDir / train / class001 / gesture.avi
... sVideoDir / val   / class249 / gesture.avi
"""

import os
import glob

import numpy as np
import pandas as pd

from video2frame import video2frame, frame2file

def videos2frames(sVideoDir:str, sFrameDir:str, nClasses = None):
    """ Extract frames from videos 
    
    Input video structure:
    ... sVideoDir / train / class001 / videoname.avi

    Output:
    ... sFrameDir / train / class001 / videoname / frames.jpg
    """

    # do not (partially) overwrite existing frame directory
    if os.path.exists(sFrameDir): raise ValueError("Frame folder " + sFrameDir + " alredy exists") 

    # get videos. Assume sVideoDir / train / class / video.avi
    dfVideos = pd.DataFrame(glob.glob(sVideoDir + "/*/*/*.avi"), columns=["sVideoPath"])
    print("Located {} videos in {}, extracting to {} ..."\
        .format(len(dfVideos), sVideoDir, sFrameDir))

    # eventually restrict to first nLabels
    if nClasses != None:
        dfVideos.loc[:,"sLabel"] = dfVideos.sVideoPath.apply(lambda s: s.split("/")[-2])
        liClasses = sorted(dfVideos.sLabel.unique())[:nClasses]
        dfVideos = dfVideos[dfVideos["sLabel"].isin(liClasses)]
        print("Using only {} videos from first {} classes".format(len(dfVideos), nClasses))

    nCounter = 0
    # loop through all videos and extract frames
    for sVideoPath in dfVideos.sVideoPath:

        # slice videos into frames with OpenCV
        arFrames = video2frame(sVideoPath, nMinDim = 240)
        
        # create diretory (assumed directories see above)
        li_sVideoPath = sVideoPath.split("/")
        if len(li_sVideoPath) < 4: raise ValueError("Video path should have min 4 components: {}".format(str(li_sVideoPath)))
        sVideoName = li_sVideoPath[-1].split(".")[0]
        sTargetDir = sFrameDir + "/" + li_sVideoPath[-3] + "/" + li_sVideoPath[-2] + "/" + sVideoName
        os.makedirs(sTargetDir, exist_ok=True)

        # write frames to .jpg files
        frame2file(arFrames, sTargetDir)            

        print("Video %5d to frames %s in %s" % (nCounter, str(arFrames.shape), sTargetDir))
        nCounter += 1      

    return


def main():
    sVideoDir = "../datasets/04-chalearn"
    sFrameDir = "06-i3d/data/frame"
    #sFlowDir =  "06-3id/data/opticalflow"

    print("Extracting ChaLearn frames and optical flow ...")
    
    videos2frames(sVideoDir, sFrameDir)

    return


if __name__ == '__main__':
    main()
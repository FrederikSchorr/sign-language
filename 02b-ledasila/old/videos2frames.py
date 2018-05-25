"""
Select video files from LedaSila database, 
extract n frames per video and save to disc

Sign language video files copyright Alpen Adria Universität Klagenfurt 
http://ledasila.aau.at

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""


import numpy as np
import pandas as pd

import os
import glob

from subprocess import call, check_output


# directories
sVideoDir = "datasets/01-ledasila"
sFrameDir = "02b-ledasila-400in20/data/1-frame"
sFeatureDir = "02b-ledasila-400in207data/2-feature"

nFramesNorm = 20 # fixed number of frames per video
nMinOccur = 18 # only consider labels with min n occurences


# In[3]:


# Read all available videos in a dataframe


# In[4]:


dfFiles = pd.DataFrame(glob.glob(sVideoDir + "/*/*.mp4"), dtype=str, columns=["sPath"])

# extract file names
dfFiles["sFile"] = dfFiles.sPath.apply(lambda s: s.split("/")[-1])

# extract word
dfFiles["sWord"] = dfFiles.sFile.apply(lambda s: s.split("-")[0])
print(dfFiles.head())

print("%d videos, with %d unique words" % (dfFiles.shape[0], dfFiles.sWord.unique().shape[0]))

# select videos with min n occurences
dfWord_freq = dfFiles.groupby("sWord").size().sort_values(ascending=False).reset_index(name="nCount")

dfWord_top = dfWord_freq.loc[dfWord_freq.nCount >= nMinOccur, :]
dfWord_top

dfVideos = pd.merge(dfFiles, dfWord_top, how="right", on="sWord")

print("Selected %d videos, %d unique words, min %d occurences" % 
      (dfVideos.shape[0], dfVideos.sWord.unique().shape[0], dfWord_top.nCount.min()))

# split train vs test data
fTest = 0.2
dfVideos["sTrain_test"] = "train"

for g in dfVideos.groupby("sWord"):
    pos = g[1].sample(frac = fTest).index
    #print(pos)
    dfVideos.loc[pos, "sTrain_test"] = "val"


dfVideos.groupby("sTrain_test").size()

#print(dfVideos.sample(5))


# extract frames from each video
nCounter = 0
for pos, seVideo in dfVideos.iterrows():
    
    # for each video create separate directory in 1-frame/train/label
    sDir = os.path.join(sFrameDir, seVideo.sTrain_test, seVideo.sWord, seVideo.sFile.split(".")[0])
    os.makedirs(sDir, exist_ok=True)
    dfVideos.loc[pos, "sPathFrames"] = sDir

    # determine length of video in sec and deduce frame rate
    fVideoSec = int(check_output(["mediainfo", '--Inform=Video;%Duration%', seVideo.sPath]))/1000.0
    #nFramesPerSec = int(ceil(nFramesNorm / fVideoSec))
    fFramesPerSec = nFramesNorm / fVideoSec

    # call ffmpeg to extract frames from videos
    sFrames = os.path.join(sDir, "frame-%03d.jpg")
    #print(sFrames)
    call(["ffmpeg", "-loglevel", "error" ,"-y", "-i", seVideo.sPath, \
            "-r", str(fFramesPerSec), "-frames", str(nFramesNorm), sFrames])

    # check right number of frames
    nFrames = len(glob.glob(sDir + "/*.jpg"))
    print("%5d | %s | %.3fsec | %.1ffps | %df" % \
        (nCounter, sDir, fVideoSec, fFramesPerSec, nFrames))
    assert(nFrames == nFramesNorm)
    nCounter += 1        

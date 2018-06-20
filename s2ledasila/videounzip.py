"""
Untar and move videos from LedaSila dataset

"""

import numpy as np
import pandas as pd

import os
import glob
import shutil
import zipfile
import warnings
import time


def unzip_sort_videos(sVideoDir:str, sVideoZip:str, nMinOccur:int = 1):
    """ Untar videos use Label information embedded in file name 
    to move videos into folders=labels
    Only for classes with nMinOccurences
    """

    print("Unzip and sort LedaSila videos from %s into %s" % (sVideoZip, sVideoDir))

    # do not (partially) overwrite existing video directory
    if os.path.exists(sVideoDir): 
        warnings.warn("Video folder " + sVideoDir + " alredy exists, extraction stopped") 
        return

    # save mother directory
    sMotherDir = os.path.abspath(sVideoDir + "/..")
    
    # unzip big zip file to tmp directory 
    sTmpDir = sMotherDir + "/tmp"
    print("Unzipping videos to %s ..." % (sTmpDir))
    if os.path.exists(sTmpDir): 
        warnings.warn("Temporary extraction folder %s already exists, extraction stopped" % (sTmpDir)) 
        return

    zip = zipfile.ZipFile(sVideoZip, "r")
    zip.extractall(sTmpDir)
    zip.close()

    # get all extracted videonames: ... sTmpDir / g / label-video.mp4
    dfFiles = pd.DataFrame(glob.glob(sTmpDir + "/*/*.mp4"), dtype=str, columns=["sVideoPath"])

    # extract labels
    dfFiles["sFileName"] = dfFiles.sVideoPath.apply(lambda s: s.split("/")[-1])
    dfFiles["sLabel"] = dfFiles.sFileName.apply(lambda s: s.split("-")[0])
    print("%d videos untarred, %d labels detected" % (len(dfFiles), len(dfFiles.sLabel.unique())))

    # restrict to classes with nMinOccurences
    dfLabel_freq = dfFiles.groupby("sLabel").size().sort_values(ascending=False).reset_index(name="nCount")
    dfLabel_top = dfLabel_freq.loc[dfLabel_freq.nCount >= nMinOccur, :]
    #print(dfLabel_top)
    dfFiles = pd.merge(dfFiles, dfLabel_top, how="right", on="sLabel")
    print("Selected %d videos from top %d labels, min %d occurences" % 
        (len(dfFiles), len(dfFiles.sLabel.unique()), dfLabel_top.nCount.min()))

    # save classes to csv file
    dfLabel_top = dfLabel_top.sort_values(by = ["sLabel"]).reset_index(drop=True)
    dfLabel_top.loc[:,"sDetail"] = " "
    sClassFile = sMotherDir + "/class.csv"
    dfLabel_top.to_csv(sClassFile, columns = ["nClass", "sClass", "nCount", "sDetail"])
    print("Classes(labels) saved to %s" % (sClassFile))

    # move video files
    nCount = 0
    for _, seVideo in dfFiles.iterrows():
        os.makedirs(sVideoDir + "/" + seVideo.sLabel, exist_ok = True)
        os.rename(seVideo.sVideoPath, sVideoDir + "/" + seVideo.sLabel + "/" + seVideo.sFileName)  
        nCount += 1

    # cleanup
    print("%d videos moved" % (nCount))
    shutil.rmtree(sTmpDir)
    print("Removed tmp folder")
    return 


def move_videos(sSourceDir, sTargetDir, fFrac = 0.2):
    """ Move fraction of the videos to another folder eg 20% from train to val 
    
    Assume folders: ... sSourceDir / class / video.mp4
    """

    # stop if new folder already exists
    assert os.path.exists(sTargetDir) == False

    sCurrentDir = os.getcwd()
    # get list of all directories = classes
    os.chdir(sSourceDir)
    liClasses = sorted(glob.glob("*"))
    print("Found %d classes in %s. Move %.0f%% of videos to %s ..." % \
        (len(liClasses), sSourceDir, fFrac*100, sTargetDir))

    # loop through directories
    for sClass in liClasses:
        os.chdir(sCurrentDir + "/" + sSourceDir + "/" + sClass)
        seVideos = pd.Series(glob.glob("*.mp4"))
        seVideosSelected = seVideos.sample(frac = fFrac)

        sTargetPath = sCurrentDir + "/" + sTargetDir + "/" + sClass
        os.makedirs(sTargetPath, exist_ok=True)

        for sVideoName in seVideosSelected:
            os.rename(sVideoName, sTargetPath + "/" + sVideoName)

        print("Class %s | %d videos: Moved %d videos" % (sClass, len(seVideos), len(seVideosSelected)))

    os.chdir(sCurrentDir)
    return 


def main():
   
    # directories
    sVideoDir = "data-set/01-ledasila/021"
    sVideoZip = "data-set/01-ledasila/_download/ledasila.zip"

    print("\nStarting LedaSila video unzip & move in directories:", os.getcwd())

    # umzip LedaSila videos and sort them in folders=label
    unzip_sort_videos(sVideoDir + "/train", sVideoZip, nMinOccur = 18)

    # move fraction of videos to another folder
    move_videos(sVideoDir + "/train", sVideoDir + "/val", fFrac = 0.2)

    return


if __name__ == '__main__':
    main()
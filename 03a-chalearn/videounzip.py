"""
Unzip and move videos from ChaLearn dataset

ChaLearn dataset:
http://chalearnlap.cvc.uab.es/dataset/21/description/

"""

import numpy as np
import pandas as pd

import os
import glob
import shutil
import zipfile

from subprocess import call, check_output
import time


def unzip_sort_videos(sVideoDir, sZipFile, sListFile):
    """ Unzip videos use Label information defined in sListFile 
    to move videos into folders=labels
    """

    # save current directory
    sCurrentDir = os.getcwd()

    # unzip videos to tmp directory 
    sTmpDir = sVideoDir + "/tmp"
    print("Unzipping videos to {} ...".format(sTmpDir))
    if os.path.exists(sTmpDir): raise ValueError("Folder {} alredy exists".format(sTmpDir)) 

    zip_ref = zipfile.ZipFile(sZipFile, "r")
    zip_ref.extractall(sTmpDir)
    zip_ref.close()
    print("{} files unzipped".format(len(glob.glob(sTmpDir + "/*/*/*.*"))))

    # read list of videos with labels    
    dfFiles = pd.read_csv(sListFile, 
        sep=" ", header=None, 
        names=["sVideoPath", "s2", "nLabel"])

    # assume sVideoPath = "train/001/M_00023.avi" and extract folders
    se_li_sVideoPath = dfFiles.sVideoPath.apply(lambda s: s.split("/"))
    dfFiles["sTrainVal"] = se_li_sVideoPath.apply(lambda s: s[0])
    dfFiles["sFileName"] = se_li_sVideoPath.apply(lambda s: s[2])

    # check target folder not yet existing
    arTrainVal = dfFiles.sTrainVal.unique()
    if len(arTrainVal) != 1: raise ValueError("sListFile supposed to contain only one train or val folder")
    sTrainValDir = arTrainVal[0]
    if os.path.exists(sTrainValDir): raise ValueError("Folder {} alredy exists".format(sVideoDir + "/" + sTrainValDir)) 

    # create new folders=classes
    seClasses = dfFiles.groupby("nLabel").size().sort_values(ascending=True)
    print("%d videos, with %d classes, occuring between %d-%d times" % \
        (len(dfFiles), len(seClasses), min(seClasses), max(seClasses)))

    for nClass, _  in seClasses.items():
        sClassDir = sVideoDir + "/" + sTrainValDir + "/" + "c{:03d}".format(nClass)
        print("Create directory", sClassDir)
        os.makedirs(sClassDir)

    # move video files (leave depth files in place)
    nCount = 0
    for pos, seVideo in dfFiles.iterrows():
        sPathNew = sTrainValDir + "/c{:03d}".format(seVideo.nLabel) + "/" + seVideo.sFileName

        if nCount % 1000 == 0:
            print ("{:5d} Move video {}".format(nCount, sPathNew))
        os.rename(sTmpDir + "/" + seVideo.sVideoPath, sVideoDir + "/" + sPathNew)  

        nCount += 1

    # cleanup
    print("{:d} videos moved".format(nCount))
    shutil.rmtree(sTmpDir)
    print("Removed tmp folder")
    return 


def move_videos(sSourceDir, sTargetDir, fFrac = 0.2):
    """ Move fraction of the videos to another folder eg 20% from train to val """

    # stop if new folder already exists
    assert os.path.exists(sTargetDir) == False

    sCurrentDir = os.getcwd()
    # get list of all directories = classes
    os.chdir(sSourceDir)
    liClasses = glob.glob("*")
    print("Found %d classes in %s. Move %.0f%% videos to %s ..." % \
        (len(liClasses), sSourceDir, fFrac*100, sTargetDir))

    # loop through directories
    for sClass in liClasses:
        os.chdir(sCurrentDir + "/" + sSourceDir + "/" + sClass)
        seVideos = pd.Series(glob.glob("*.avi"))
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
    sVideoDir = "datasets/04-chalearn"
    sTrainZip = sVideoDir + "/_zip/train.zip"
    sTrainList = sVideoDir + "/_zip/train.txt"
    sLogDir = "03a-chalearn/log"

    print("\nStarting ChaLearn video unzip & move in directory:", os.getcwd())

    # unzip ChaLearn videos and sort them in folders=label
    #unzip_sort_videos(sVideoDir, sTrainZip, sTrainList)

    # move fraction of videos to another folder
    move_videos(sVideoDir + "/train", sVideoDir + "/val", fFrac = 0.2)

    return


if __name__ == '__main__':
    main()
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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from inceptionfeatures import InceptionV3_features

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger


def copyvideos(sListFile, sVideoDir, sNewFolder, nLabels = None):
    """ Copy videos from original ChaLearn folder structure defined in sListFile
    to folders=labels
    """
    # stop if new folder already exists
    assert os.path.exists(sVideoDir + "/" + sNewFolder) == False
    
    dfFiles = pd.read_csv(sListFile, 
        sep=" ", header=None, 
        names=["sVideoPath", "s2", "nLabel"])

    # reduce sample for testing purpose
    if nLabels != None: dfFiles = dfFiles.loc[dfFiles.nLabel <= nLabels, :].copy()

    # create new folders=labels
    seLabels = dfFiles.groupby("nLabel").size().sort_values(ascending=True)
    print("%d videos, with %d labels, occuring between %d-%d times" % \
        (len(dfFiles), len(seLabels), min(seLabels), max(seLabels)))
    for nLabel, nOcc  in seLabels.items():
        sDir = sVideoDir + "/" + sNewFolder + "/" + "{:03d}".format(nLabel)
        print("Create directory", sDir)
        os.makedirs(sDir)

    # copy video files
    nCount = 0
    for pos, seVideo in dfFiles.iterrows():
        sFileName = seVideo.sVideoPath.split("/")[2]
        sPathNew = sNewFolder + "/" + "{:03d}".format(seVideo.nLabel) + "/" + sFileName

        if nCount % 1000 == 0:
            print ("{:5d} Copy {:s} to {:s}".format(nCount, seVideo.sVideoPath, sPathNew))
        shutil.copy(sVideoDir + "/" + seVideo.sVideoPath, sVideoDir + "/" + sPathNew)  

        seVideo.sVideoPath = sPathNew
        nCount += 1

    print("{:d} videos copied".format(nCount))
    return dfFiles


def prepFileList(sListFile, sLogDir, fVal = 0.2, nLabels = None):
    """ Attention: this function assumes original ChaLearn folder structure """
    dfFiles = pd.read_csv(sListFile, 
        sep=" ", header=None, 
        names=["sVideoPath", "s2", "nLabel"])

    # reduce sample for testing purpose
    if nLabels != None: dfFiles = dfFiles.loc[dfFiles.nLabel <= nLabels, :].copy()

    seLabels = dfFiles.groupby("nLabel").size().sort_values(ascending=False)
    print("%d videos, with %d labels, occuring between %d-%d times" % \
        (len(dfFiles), len(seLabels), min(seLabels), max(seLabels)))

    # split train vs val data
    dfFiles.loc[:,"sTrain_val"] = "train"
    for g in dfFiles.groupby("nLabel"):
        pos = g[1].sample(frac = fVal).index
        dfFiles.loc[pos, "sTrain_val"] = "val"
    print(dfFiles.groupby("sTrain_val").size())

    # target directories
    seDir3 = dfFiles.sVideoPath.apply(lambda s: s.split("/")[2])
    seDir3 = seDir3.apply(lambda s: s.split(".")[0])
    dfFiles.loc[:, "sFrameDir"] = dfFiles.sTrain_val + "/" + dfFiles.nLabel.astype("str") + "/" + seDir3

    sFilesListPath = sLogDir + "/" + time.strftime("%Y%m%d-%H%M") + "-list.csv"
    print("Save list to %s" % sFilesListPath)
    dfFiles.to_csv(sFilesListPath)
    return dfFiles


def video2frames(dfFiles, sVideoDir, sFrameDir, nFramesNorm = 20):
    """ Extract frames from videos """
    
    # prepare frame counting
    dfFiles.loc[:, "nFrames"] = 0
    nFramesTotal = len(glob.glob(os.path.join(sFrameDir, "*/*/*/*.jpg")))
    assert(nFramesTotal == 0) # risk that video is extracted into train PLUS val directory
    nCounter = 0

    # extract frames from each video
    print("Extract frames from %d videos (%d frames) ..." % (len(dfFiles), nFramesNorm))
    for pos, seVideo in dfFiles.iterrows():
        
        # source path
        sVideoPath = os.path.join(sVideoDir, seVideo.sVideoPath)

        # create frame directory for each video
        sDir = os.path.join(sFrameDir, seVideo.sFrameDir)
        os.makedirs(sDir, exist_ok=True)
        
        # determine length of video in sec and deduce frame rate
        fVideoSec = int(check_output(["mediainfo", '--Inform=Video;%Duration%', sVideoPath]))/1000.0
        #nFramesPerSec = int(ceil(nFramesNorm / fVideoSec))
        fFramesPerSec = nFramesNorm / fVideoSec

        # call ffmpeg to extract frames from videos
        sFrames = os.path.join(sDir, "frame-%03d.jpg")
        call(["ffmpeg", "-loglevel", "error" ,"-y", "-i", sVideoPath, \
            "-r", str(fFramesPerSec), "-frames", str(nFramesNorm), sFrames])
            
        # check right number of frames
        nFrames = len(glob.glob(sDir + "/*.jpg"))
        dfFiles.loc[pos, "nFrames"] = nFrames
        print("%5d | %s => %s | %.3fsec | %.1ffps | %df" % \
            (nCounter, sVideoPath, sDir, fVideoSec, fFramesPerSec, nFrames))
        assert(nFrames == nFramesNorm)
        nCounter += 1

    # check number of created frames
    nFramesTotal = len(glob.glob(os.path.join(sFrameDir, "*/*/*/*.jpg"))) - nFramesTotal
    print("%d frames extracted from %d videos" % (nFramesTotal, len(dfFiles)))

    return


def frames2features(sFrameDir, sFeatureDir, nFramesNorm, nLabels=None):
    """ Use pretrained CNN to calculate features from video-frames """

    sCurrentDir = os.getcwd()
    # get list of directories with frames
    os.chdir(sFrameDir)
    dfVideos = pd.DataFrame(glob.glob("train/*/*"), dtype=str, columns=["sFrameDir"])
    dfVideos = pd.concat([dfVideos,
               pd.DataFrame(glob.glob("val/*/*"), dtype=str, columns=["sFrameDir"])])
    os.chdir(sCurrentDir)
    print("Found %d directories with frames" % len(dfVideos))

    # eventually restrict to first nLabels
    if nLabels != None:
        dfVideos.loc[:,"sLabel"] = dfVideos.sFrameDir.apply(lambda s: s.split("/")[-2])
        liLabels = sorted(dfVideos.sLabel.unique())[:nLabels]
        dfVideos = dfVideos[dfVideos["sLabel"].isin(liLabels)]
        print("Extracting features from %d directories (%d Labels)" % (len(dfVideos), nLabels))

    # get the InceptionV3 model
    cnn = InceptionV3_features()

    # feed all frames into cnn, save results in file per video
    nCounter = 0
    for _, seVideo in dfVideos.iterrows():

        # save resulting list of features in file in "data/2-feature/train/label/video.npy"
        sFeaturePath = os.path.join(sFeatureDir, seVideo.sFrameDir + ".npy")
        if (os.path.exists(sFeaturePath)):
            # if feature already etracted, skip
            print("%5d | Features %s already exist" % (nCounter, sFeaturePath))
            continue
        os.makedirs(os.path.dirname(sFeaturePath), exist_ok=True)

        # retrieve frame files - in ascending order
        sFrames = sorted(glob.glob(os.path.join(sFrameDir, seVideo.sFrameDir, "*.jpg")))
        print("%5d | Extracting features from %d frames to %s" % (nCounter, len(sFrames), sFeaturePath))
        assert(len(sFrames) == nFramesNorm)

        # run cnn on each frame, collect the resulting features
        liFeatures = []
        for sFrame in sFrames:
            features = cnn.extract(sFrame)
            liFeatures.append(features)   
        
        np.save(sFeaturePath, liFeatures)
        nCounter += 1
    return


class Features():
    """ Read features from disc, check size and save in arrays """

    def __init__(self, sPath, nFramesNorm, nFeatureLength):
        
        # retrieve all feature files. Expected folder structure .../train/label/feature.npy
        self.liPaths = glob.glob(os.path.join(sPath, "*", "*.npy"))
        self.nSamples = len(self.liPaths)
        if self.nSamples == 0:
            print("Error: found no feature files in %s" % sPath)
        print("Load %d samples from %s ..." % (self.nSamples, sPath))

        self.arFeatures = np.zeros((self.nSamples, nFramesNorm, nFeatureLength))
        
        # loop through all feature files
        self.liLabels = [] 
        for i in range(self.nSamples):
            
            # Get the sequence from disc
            sFile = self.liPaths[i]
            arF = np.load(sFile)
            if (arF.shape != (nFramesNorm, nFeatureLength)):
                print("Error: %s has wrong shape %s" % (sFile, arF.shape))
            self.arFeatures[i] = arF
            
            # Extract the label from path
            sLabel = sFile.split("/")[-2]
            self.liLabels.append(sLabel)
            
        # labels
        self.liClasses = sorted(np.unique(self.liLabels))
        self.nClasses = len(self.liClasses)
        
        # one hot encode labels
        label_encoder = LabelEncoder()
        arLabelsNumerical = label_encoder.fit_transform(self.liLabels).reshape(-1,1)
        onehot_encoder = OneHotEncoder(sparse=False)
        self.arLabelsOneHot = onehot_encoder.fit_transform(arLabelsNumerical)
        
        print("Loaded %d samples from %d classes" % (self.nSamples, self.nClasses))
        
        return

        
def train(sFeatureDir, sModelDir, sModelSaved, sLogDir, 
          nFramesNorm = 20, nFeatureLength = 2048, nBatchSize=16, nEpoch=100, fLearn=1e-4):

    # Load features
    oFeatureTrain = Features(sFeatureDir + "/train", nFramesNorm, nFeatureLength)
    oFeatureVal = Features(sFeatureDir + "/val", nFramesNorm, nFeatureLength)
    assert(oFeatureTrain.liLabels == oFeatureVal.liLabels)

    # Build a simple LSTM network. We pass the extracted features from
    # our CNN to this model     
    
    if sModelSaved == None:
        # Build new model
        keModel = Sequential()
        keModel.add(LSTM(1024, return_sequences=True,
                        input_shape=(nFramesNorm, nFeatureLength),
                        dropout=0.5))
        keModel.add(LSTM(1024, return_sequences=False, dropout=0.5))
        #keModel.add(Dense(256, activation='relu'))
        #keModel.add(Dropout(0.5))
        keModel.add(Dense(oFeatureTrain.nLabels, activation='softmax'))

        # Now compile the network.
        optimizer = Adam(lr=fLearn)
        keModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    else:
        # load model from file
        print("Loading saved LSTM neural network %s ..." % sModelSaved)
        keModel = load_model(sModelSaved)
        assert(keModel.input_shape == (None, nFramesNorm, nFeatureLength))

    keModel.summary()
    # Helper: TensorBoard
    #tb = TensorBoard(log_dir=os.path.join("../data/90-logs", model))

    # Helper: Stop when we stop learning.
    #early_stopper = EarlyStopping(patience=5)

    # Helper: Save results
    os.makedirs(sLogDir, exist_ok=True)
    sLog = time.strftime("%Y%m%d-%H%M") + "-lstm-" + \
        str(oFeatureTrain.nSamples + oFeatureVal.nSamples) + "in" + str(oFeatureTrain.nLabels)
    csv_logger = CSVLogger(os.path.join(sLogDir, sLog + '.acc'))

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    checkpointer = ModelCheckpoint(
        filepath = sModelDir + "/" + sLog + "-best.h5",
        verbose = 1, save_best_only = True)
 
    # Fit!
    keModel.fit(
        oFeatureTrain.arFeatures,
        oFeatureTrain.arLabelsOneHot,
        batch_size = nBatchSize,
        epochs = nEpoch,
        verbose = 1,
        shuffle=True,
        validation_data=(oFeatureVal.arFeatures, oFeatureVal.arLabelsOneHot),
        #callbacks=[tb, early_stopper, csv_logger, checkpointer],
        callbacks=[csv_logger, checkpointer]
    )    
    
    # save model
    keModel.save(sModelDir + "/" + sLog + "-last.h5")

    return

def evaluate(sFeatureDir, sModelSaved, nFramesNorm, nFeatureLength):    
    """ evaluate all features in given directory on saved model """

    # Load features
    oFeatures = Features(sFeatureDir, nFramesNorm, nFeatureLength)

    # load model from file
    print("Loading saved LSTM neural network %s ..." % sModelSaved)
    keModel = load_model(sModelSaved)
    assert(keModel.input_shape == (None, nFramesNorm, nFeatureLength))

    liResult = keModel.evaluate(
        oFeatures.arFeatures,
        oFeatures.arLabelsOneHot,
        batch_size = None,
        verbose = 1)
    
    print(keModel.metrics_names)
    print(liResult)

    return


def main():
   
    # directories
    sVideoDir = "datasets/04-chalearn"
    sListFile = sVideoDir + "/train_list.txt"
    sFrameDir = "03a-chalearn/data/frame"
    sFeatureDir = "03a-chalearn/data/feature"
    sModelDir = "03a-chalearn/model"
    sLogDir = "03a-chalearn/log"

    nLabels = None
    nFramesNorm = 20 # number of frames per video for feature calculation
    nFeatureLength = 2048 # output features from CNN

    print("\nStarting ChaLearn extraction & train in directory:", os.getcwd())

    # copy videos from original ChaLearn folders to folders=label
    #copyvideos(sListFile, sVideoDir, "train_l", None)

    # extract frames from videos
    #dfFiles = prepFileList(sListFile, sLogDir, fVal = 0.2, nLabels = nLabels)
    #video2frames(dfFiles, sVideoDir, sFrameDir, nFramesNorm)
    
    # calculate features from frames
    #frames2features(sFrameDir, sFeatureDir, nFramesNorm, nLabels = nLabels)

    # train the LSTM network
    sModelSaved = sModelDir + "/20180525-1033-lstm-35878in249-best.h5"
    #train(sFeatureDir, sModelDir, None, sLogDir, 
    #      nFramesNorm, nFeatureLength, nBatchSize=256, nEpoch=100, fLearn=1e-3)

    # evaluate features on LSTM
    evaluate(sFeatureDir + "/val", sModelSaved, nFramesNorm, nFeatureLength)

if __name__ == '__main__':
    main()
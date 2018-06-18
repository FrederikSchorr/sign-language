"""
Classify human motion videos from LedaSila dataset

Sign language video files copyright Alpen Adria Universität Klagenfurt 
http://ledasila.aau.at

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""

import numpy as np
import pandas as pd

from math import ceil

import os
import glob

from subprocess import call, check_output
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from inceptionfeatures import InceptionV3_features

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger


def video2frames(sVideoDir, sFrameDir, sFeatureDir, nFramesNorm, nMinOccur, fVal):
      
    # load all video file names
    dfFiles = pd.DataFrame(glob.glob(sVideoDir + "/*/*.mp4"), dtype=str, columns=["sPath"])

    # extract file names
    dfFiles["sFile"] = dfFiles.sPath.apply(lambda s: s.split("/")[-1])

    # extract word
    dfFiles["sWord"] = dfFiles.sFile.apply(lambda s: s.split("-")[0])
    print("%d videos, with %d unique words" % (dfFiles.shape[0], dfFiles.sWord.unique().shape[0]))

    # select videos with min n occurences
    dfWord_freq = dfFiles.groupby("sWord").size().sort_values(ascending=False).reset_index(name="nCount")
    dfWord_top = dfWord_freq.loc[dfWord_freq.nCount >= nMinOccur, :]
    #print(dfWord_top)

    dfVideos = pd.merge(dfFiles, dfWord_top, how="right", on="sWord")
    print("Selected %d videos, %d unique words, min %d occurences" % 
        (dfVideos.shape[0], dfVideos.sWord.unique().shape[0], dfWord_top.nCount.min()))

    # split train vs test data
    dfVideos["sTrain_val"] = "train"

    for g in dfVideos.groupby("sWord"):
        pos = g[1].sample(frac = fVal).index
        #print(pos)
        dfVideos.loc[pos, "sTrain_val"] = "val"

    print(dfVideos.groupby("sTrain_val").size())

    # check for already existing image frames
    nFramesTotal = len(glob.glob(os.path.join(sFrameDir, "*/*/*/*.jpg")))
    assert(nFramesTotal == 0) # risk that video is extracted into train PLUS val directory

    # extract frames from each video
    nCounter = 0
    for pos, seVideo in dfVideos.iterrows():
        
        # for each video create separate directory in frame/train/label
        sDir = os.path.join(sFrameDir, seVideo.sTrain_val, seVideo.sWord, seVideo.sFile.split(".")[0])
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
        liLabels = []
        for i in range(self.nSamples):
            
            # Get the sequence from disc
            sFile = self.liPaths[i]
            arF = np.load(sFile)
            if (arF.shape != (nFramesNorm, nFeatureLength)):
                print("Error: %s has wrong shape %s" % (sFile, arF.shape))
            self.arFeatures[i] = arF
            
            # Extract the label from path
            sLabel = sFile.split("/")[-2]
            liLabels.append(sLabel)
            
        # labels
        self.liLabels = sorted(np.unique(liLabels))
        self.nLabels = len(self.liLabels)
        
        # one hot encode labels
        label_encoder = LabelEncoder()
        arLabelsNumerical = label_encoder.fit_transform(liLabels).reshape(-1,1)
        onehot_encoder = OneHotEncoder(sparse=False)
        self.arLabelsOneHot = onehot_encoder.fit_transform(arLabelsNumerical)
        
        print("Loaded %d samples with %d labels" % (self.nSamples, self.nLabels))
        
        return

        
def train(sFeatureDir, sModelDir, sModelSaved, sLogDir, 
          nFramesNorm = 20, nFeatureLength = 2048, nBatchSize=16, nEpoch=100, fLearn=0.0001, fDropout=0.5):

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
                        dropout=fDropout))
        keModel.add(LSTM(1024, return_sequences=False, dropout=fDropout))
        #keModel.add(Dense(256, activation='relu'))
        #keModel.add(Dropout(fDropout))
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
    #os.makedirs(sModelDir, exist_ok=True)
    #checkpointer = ModelCheckpoint(
    #    filepath = sModelDir + "/" + sLog + "-best.h5",
    #    verbose = 1, save_best_only = True)
 
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
        callbacks=[csv_logger]
    )    
    
    # save model
    keModel.save(sModelDir + "/" + sLog + "-last.h5")


def main():
   
    # directories
    #sVideoDir = "datasets/01-ledasila"  
    #sFrameDir = "02b-ledasila/data/frame"
    sFeatureDir = "02b-ledasila/data/0440in21/feature"
    sModelDir = "02b-ledasila/model"
    sLogDir = "02b-ledasila/log"

    #nMinOccur = 18 # only take labels with min n occurrences in dataset
    #nLabels = None
    nFramesNorm = 20 # number of frames per video for feature calculation
    nFeatureLength = 2048 # output features from CNN

    print("\nStarting LedaSila extraction & train in directory:", os.getcwd())
    
    # Convert videos to frames
    #video2frames(sVideoDir, sFrameDir, sFeatureDir, nFramesNorm, nMinOccur, fVal = 0.2)

    # calculate features from frames
    #frames2features(sFrameDir, sFeatureDir, nFramesNorm, nLabels)

    # train the LSTM network
    train(sFeatureDir, sModelDir, None, sLogDir, 
          nFramesNorm, nFeatureLength, nBatchSize=16, nEpoch=500, fLearn=1e-4, fDropout=0.5)

if __name__ == '__main__':
    main()
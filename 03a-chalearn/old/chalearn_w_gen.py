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

from subprocess import call, check_output
import time

from inceptionfeatures import InceptionV3_features

from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from featuregenerator import FeatureGenerator


def prepFileList(sListFile, sLogDir, fVal = 0.2, nLabels = None):
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


def train(sFeatureDir, sModelDir, sLogDir, nFramesNorm = 20, nFeatureLength = 2048,
          saved_model=None, nBatchSize=16, nEpoch=100):

    # Get generators.
    oFeatureGenerator = FeatureGenerator(sFeatureDir, "train", "val")
    train_generator = oFeatureGenerator.frame_generator("train", nBatchSize, nFramesNorm, nFeatureLength)
    val_generator = oFeatureGenerator.frame_generator("val", nBatchSize, nFramesNorm, nFeatureLength)

    # Build a simple LSTM network. We pass the extracted features from
    # our CNN to this model     
    keModel = Sequential()
    keModel.add(LSTM(1024, return_sequences=False,
                    input_shape=(nFramesNorm, nFeatureLength),
                    dropout=0.5))
    keModel.add(Dense(256, activation='relu'))
    keModel.add(Dropout(0.5))
    keModel.add(Dense(oFeatureGenerator.nClasses, activation='softmax'))
    keModel.summary()

    # Now compile the network.
    optimizer = Adam(lr=1e-6, decay=1e-7)
    keModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Helper: TensorBoard
    #tb = TensorBoard(log_dir=os.path.join("../data/90-logs", model))

    # Helper: Stop when we stop learning.
    #early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    sLog = time.strftime("%Y%m%d-%H%M") + "-lstm-" + \
        str(oFeatureGenerator.nTrain + oFeatureGenerator.nTest) + "in" + str(oFeatureGenerator.nClasses)
    csv_logger = CSVLogger(os.path.join(sLogDir, sLog + '.log'))

    # Helper: Save the model.
    #checkpointer = ModelCheckpoint(
    #    filepath=os.path.join("../data/30-checkpoints", sLog + "-{epoch:03d}.hdf5"),
    #    verbose=1, save_best_only=True)
 
    # Fit!
    keModel.fit_generator(
        generator=train_generator,
        steps_per_epoch=oFeatureGenerator.nTrain // nBatchSize,
        epochs=nEpoch,
        verbose=1,
        #callbacks=[tb, early_stopper, csv_logger, checkpointer],
        callbacks=[csv_logger],
        validation_data=val_generator,
        validation_steps=40,
        workers=4
    )    
    keModel.save(sModelDir + "/" + sLog + ".h5")


def main():
   
    # directories
    sVideoDir = "datasets/04-chalearn"
    sListFile = sVideoDir + "/train_list.txt"
    sFrameDir = "03a-chalearn/data/1-frame"
    sFeatureDir = "03a-chalearn/data/2-feature"
    sModelDir = "03a-chalearn/data/3-model"
    sLogDir = "03a-chalearn/data/9-log"

    nLabels = 10
    nFramesNorm = 20 # number of frames per video for feature calculation
    nFeatureLength = 2048 # output features from CNN

    print("\nStarting ChaLearn extraction & train in directory:", os.getcwd())

    # extract frames from videos
    #dfFiles = prepFileList(sListFile, sLogDir, fVal = 0.2, nLabels = nLabels)
    #video2frames(dfFiles, sVideoDir, sFrameDir, nFramesNorm)
    
    # calculate features from frames
    #frames2features(sFrameDir, sFeatureDir, nFramesNorm, nLabels = nLabels)

    # train the LSTM network
    train(sFeatureDir, sModelDir, sLogDir, nFramesNorm, nFeatureLength = nFeatureLength,
          saved_model=None, nBatchSize=8, nEpoch=500)

if __name__ == '__main__':
    main()
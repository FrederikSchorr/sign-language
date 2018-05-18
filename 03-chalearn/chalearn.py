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

from datagenerator import DataSet


def video2frames(sListFile, sVideoDir, sFrameDir, fVal = 0.2, nFramesNorm = 20):
    """ Extract frames from videos """
    dfFiles = pd.read_csv(
        os.path.join(sVideoDir, sListFile), 
        sep=" ", header=None, 
        names=["sVideoPath", "s2", "nLabel"])

    seLabels = dfFiles.groupby("nLabel").size().sort_values(ascending=False)
    print("%d videos, with %d labels, occuring between %d-%d times" % \
        (len(dfFiles), len(seLabels), min(seLabels), max(seLabels)))

    # split train vs val data
    dfFiles["sTrain_val"] = "train"
    for g in dfFiles.groupby("nLabel"):
        pos = g[1].sample(frac = fVal).index
        dfFiles.loc[pos, "sTrain_val"] = "val"
    print(dfFiles.groupby("sTrain_val").size())

    # reduce sample for testing purpose
    dfFiles = dfFiles.query("nLabel <= 3")

    # target directories
    seDir3 = dfFiles.sVideoPath.apply(lambda s: s.split("/")[2])
    seDir3 = seDir3.apply(lambda s: s.split(".")[0])
    dfFiles["sFrameDir"] = dfFiles.sTrain_val + "/" + dfFiles.nLabel.astype("str") + "/" + seDir3
    
    # prepare frame counting
    dfFiles["nFrames"] = 0
    nFramesTotal = len(glob.glob(os.path.join(sFrameDir, "*/*/*/*.jpg")))
    nCounter = 0

    # extract frames from each video
    print("Extract frames from %d videos (%d frames) ..." % (len(dfFiles), nFramesNorm))
    for pos, seVideo in dfFiles.iterrows():
        
        # source path
        sVideoPath = os.path.join(sVideoDir, seVideo.sVideoPath)

        # create frame directory for each video
        sDir = os.path.join(sFrameDir, seVideo.sFrameDir)
        os.makedirs(sDir)
        
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

    return dfFiles


def frames2features(dfFiles, sFrameDir, sFeatureDir):
    """ Use pretrained CNN to calculate features from video-frames """

    # get the InceptionV3 model
    cnn = InceptionV3_features()

    # feed all frames into cnn, save results in file per video
    nCounter = 0
    for pos, seVideo in dfFiles.iterrows():

        # save resulting list of features in file in "data/2-feature/train/label/video.npy"
        sFeaturePath = os.path.join(sFeatureDir, seVideo.sFrameDir + ".npy")
        print("%5d | Extracting features to %s" % (nCounter, sFeaturePath))
        os.makedirs(os.path.dirname(sFeaturePath), exist_ok=True)

        # retrieve frame files
        sFrames = glob.glob(os.path.join(sFrameDir, seVideo.sFrameDir, "*.jpg"))
        
        # run cnn on each frame, collect the resulting features
        liFeatures = []
        for sFrame in sFrames:
            features = cnn.extract(sFrame)
            liFeatures.append(features)   
        
        np.save(sFeaturePath, liFeatures)
        nCounter += 1
    return


def train(sFeatureDir, sModelDir, sLogDir, sModel, nFramesNorm = 20, nFeatureLength = 2048,
          saved_model=None, nBatchSize=16, nEpoch=100):

    # Get generators.
    data = DataSet(sFeatureDir, "train", "val")
    train_generator = data.frame_generator("train", nBatchSize)
    val_generator = data.frame_generator("val", nBatchSize)

    # Build a simple LSTM network. We pass the extracted features from
    # our CNN to this model     
    keModel = Sequential()
    keModel.add(LSTM(1024, return_sequences=False,
                    input_shape=(nFramesNorm, nFeatureLength),
                    dropout=0.5))
    keModel.add(Dense(256, activation='relu'))
    keModel.add(Dropout(0.5))
    keModel.add(Dense(data.nClasses, activation='softmax'))
    keModel.summary()

    # Now compile the network.
    optimizer = Adam(lr=1e-5, decay=1e-6)
    keModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Helper: TensorBoard
    #tb = TensorBoard(log_dir=os.path.join("../data/90-logs", model))

    # Helper: Stop when we stop learning.
    #early_stopper = EarlyStopping(patience=5)

    # Helper: Save results.
    sLog = time.strftime("%Y%m%d-%H%M") + "-" + sModel + "-" + str(data.nTrain) + "in" + str(data.nClasses)
    csv_logger = CSVLogger(os.path.join(sLogDir, sLog + '.log'))

    # Helper: Save the model.
    #checkpointer = ModelCheckpoint(
    #    filepath=os.path.join("../data/30-checkpoints", sLog + "-{epoch:03d}.hdf5"),
    #    verbose=1, save_best_only=True)
 
    # Fit!
    keModel.fit_generator(
        generator=train_generator,
        steps_per_epoch=data.nTrain // nBatchSize,
        epochs=nEpoch,
        verbose=1,
        #callbacks=[tb, early_stopper, csv_logger, checkpointer],
        callbacks=[csv_logger],
        validation_data=val_generator,
        validation_steps=40,
        workers=4
    )    


def main():
   
    # directories
    sVideoDir = "datasets/04-chalearn"
    sListFile = "train_list.txt"
    sFrameDir = "03-chalearn/data/1-frame"
    sFeatureDir = "03-chalearn/data/2-feature"
    sModelDir = "03-chalearn/data/3-model"
    sLogDir = "03-chalearn/data/9-log/"

    fVal = 0.2 # percentage of validation set
    nFramesNorm = 20 # number of frames per video for feature calculation

    sModel = 'lstm'
    nFeatureLength = 2048 # output features from CNN
    nBatchSize = 16
    nEpoch = 3

    print("Current directory:", os.getcwd())

    # extract frames from videos
    dfFiles = video2frames(sListFile, sVideoDir, sFrameDir, fVal, nFramesNorm)
    sListPath = sLogDir + "/" + time.strftime("%Y%m%d-%H%M") + "-list.csv"
    print("Save list to %s" % sListPath)
    dfFiles.to_csv(sListPath)

    # calculate features from frames
    #dfFiles = pd.read_csv(sLogDir + "/20180518-0743-list.csv")
    frames2features(dfFiles, sFrameDir, sFeatureDir)

    # train the LSTM network
    train(sFeatureDir, sModelDir, sLogDir, sModel, nFramesNorm, nFeatureLength = nFeatureLength,
          saved_model=None, nBatchSize=nBatchSize, nEpoch=nEpoch)

if __name__ == '__main__':
    main()
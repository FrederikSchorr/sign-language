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

from videounzip import unzip_sort_videos

from deeplearning import ConvNet, RecurrentNet, VideoClasses, VideoFeatures

#from keras.models import Model, Sequential, load_model
#from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

import keras.preprocessing


def frames2features(sFrameDir, sFeatureDir, oCNN, nFramesNorm, nClasses=None):
    """ Use pretrained CNN to calculate features from video-frames """

    # do not (partially) overwrite existing feature directory
    if os.path.exists(sFeatureDir): raise ValueError("Folder %s alredy exists" % sFeatureDir) 

    sCurrentDir = os.getcwd()
    # get list of directories with frames: ... / sFrameDir/train/class/videodir/frames.jpg
    os.chdir(sFrameDir)
    dfVideos = pd.DataFrame(glob.glob("*/*/*"), dtype=str, columns=["sFrameDir"])
    os.chdir(sCurrentDir)
    print("Found %d directories=videos with frames" % len(dfVideos))

    # eventually restrict to first nLabels
    if nClasses != None:
        dfVideos.loc[:,"sLabel"] = dfVideos.sFrameDir.apply(lambda s: s.split("/")[-2])
        liClasses = sorted(dfVideos.sLabel.unique())[:nClasses]
        dfVideos = dfVideos[dfVideos["sLabel"].isin(liClasses)]
        print("Using only %d directories from %d classes" % (len(dfVideos), nClasses))

    # loop over all videos-directories. 
    # Feed all frames into ConvNet, save results in file per video
    nCounter = 0
    for _, seVideo in dfVideos.iterrows():

        # save resulting list of features in file in eg "data/feature/train/label/video.npy"
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

        # assemble array with all images
        liFrames = []
        for sFrame in sFrames:
            
            # load frame
            pilFrame = keras.preprocessing.image.load_img(sFrame, target_size=oCNN.tuHeightWidth)
            arFrame = keras.preprocessing.image.img_to_array(pilFrame)
            liFrames.append(arFrame)  
        
        # predict features with ConvNet
        arFrames = oCNN.preprocess_input(np.array(liFrames))
        arFeatures = oCNN.keModel.predict(arFrames)
       
        # flatten the features
        arFeatures = arFeatures.reshape(nFramesNorm, -1)

        # save to file
        np.save(sFeaturePath, arFeatures)
        nCounter += 1
    return

        
def train(sFeatureDir, sModelDir, sLogDir, oRNN,
          nBatchSize=16, nEpoch=100, fLearn=1e-4):
    print("\nTrain LSTM ...")

    # Load features
    oFeatureTrain = VideoFeatures(sFeatureDir + "/train", 
        oRNN.nFramesNorm, oRNN.nFeatureLength, oRNN.oClasses.liClasses)
    oFeatureVal = VideoFeatures(sFeatureDir + "/val", 
        oRNN.nFramesNorm, oRNN.nFeatureLength, oRNN.oClasses.liClasses)

    # prep logging
    os.makedirs(sLogDir, exist_ok=True)
    sLog = time.strftime("%Y%m%d-%H%M") + "-lstm-" + \
        str(oFeatureTrain.nSamples + oFeatureVal.nSamples) + "in" + str(oFeatureTrain.nClasses)
    
    # Helper: TensorBoard
    #tb = TensorBoard(log_dir=os.path.join("../data/90-logs", model))

    # Helper: Stop when we stop learning.
    #early_stopper = EarlyStopping(patience=5)

    # Helper: Save results
    csv_logger = CSVLogger(os.path.join(sLogDir, sLog + '-acc.csv'))

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    checkpointer = ModelCheckpoint(
        filepath = sModelDir + "/" + sLog + "-best.h5",
        verbose = 1, save_best_only = True)
 
    # Fit!
    oRNN.keModel.fit(
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
    sModelSaved = sModelDir + "/" + sLog + "-last.h5"
    oRNN.keModel.save(sModelSaved)

    return oRNN


def evaluate(sFeatureDir, oRNN):    
    """ evaluate all features in given directory on saved model """
    print("\nEvaluate LSTM ...")

    # Load features
    oFeatures = VideoFeatures(sFeatureDir, 
        oRNN.nFramesNorm, oRNN.nFeatureLength, oRNN.oClasses.liClasses)

    # evaluate 
    liResult = oRNN.keModel.evaluate(
        oFeatures.arFeatures,
        oFeatures.arLabelsOneHot,
        batch_size = None,
        verbose = 1)
    
    print(oRNN.keModel.metrics_names)
    print(liResult)

    return


def predict(sFeatureDir, oRNN):    
    """ predict class for all features in given directory on saved model """
    
    print("\nPredict features on LSTM ...")
    
    # Load video features
    oFeatures = VideoFeatures(sFeatureDir, 
        oRNN.nFramesNorm, oRNN.nFeatureLength, oRNN.oClasses.liClasses)

    # predict
    arProba = oRNN.keModel.predict(
        oFeatures.arFeatures, 
        batch_size = None, 
        verbose = 1
    )
    arPred = arProba.argmax(axis=1)
    
    # compare
    #print("Groundtruth:", oFeatures.liLabels)
    #print("Predicted:  ", arPred)
    #print("Predicted:  ", dfClass.sClass[arPred])
    print("Accuracy: {:.3f}".format(np.mean(oFeatures.liLabels == oRNN.oClasses.dfClass.loc[arPred, "sClass"])))

    return


def main():
   
    nClasses = 20
    nFramesNorm = 20 # number of frames per video for feature calculation

    # directories
    sClassFile = "data-set/04-chalearn/class.csv"
    #sVideoDir = "data-set/04-chalearn"
    sFrameDir = "data-temp/04-chalearn/%03d-oflow-20"%(nClasses)
    sFeatureDir = "data-temp/04-chalearn/%03d-oflow-20-mobilenet"%(nClasses)
    sModelDir = "model"
    sLogDir = "log"

    #sModelSaved = sModelDir + "/20180612-0740-lstm-13in249-last.h5"

    print("\nStarting ChaLearn extraction & train in directory:", os.getcwd())

    # unzip ChaLearn videos and sort them in folders=label
    #unzip_sort_videos(sVideoDir, sVideoDir + "/_zip/train.zip", sVideoDir + "/_zip/train.txt")
    #unzip_sort_videos(sVideoDir, sVideoDir + "/_zip/val.zip", sVideoDir + "/_zip/val.txt")

    # extract frames from videos
    #video2frames(sVideoDir, sFrameDir, nFramesNorm, nClasses = None)
    
    # calculate features from frames
    oCNN = ConvNet("mobilenet")
    oCNN.load_model()
    frames2features(sFrameDir, sFeatureDir, oCNN, nFramesNorm, nClasses)

    # train the LSTM network
    oClasses = VideoClasses(sClassFile)
    oRNN = RecurrentNet("lstm", nFramesNorm, oCNN.nOutputFeatures, oClasses)
    oRNN.build_compile(fLearn=1e-3)
    #oRNN.load_model(sModelSaved)
    oRNN = train(sFeatureDir, sModelDir, sLogDir, oRNN, 
        nBatchSize=256, nEpoch=5)

    # evaluate features on LSTM
    #evaluate(sFeatureDir + "/val", oRNN)

    # predict labels from features
    predict(sFeatureDir + "/val", oRNN)

if __name__ == '__main__':
    main()
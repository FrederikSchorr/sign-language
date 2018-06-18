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

import warnings

from subprocess import call, check_output
import time

from s3chalearn.videounzip import unzip_sort_videos
from s3chalearn.deeplearning import ConvNet, RecurrentNet, VideoClasses, VideoFeatures
from s3chalearn.feature import frames2features

#from keras.models import Model, Sequential, load_model
#from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

import keras.preprocessing

        
def train(sFeatureDir, sModelDir, sLogPath, oRNN,
          nBatchSize=16, nEpoch=100, fLearn=1e-4):
    print("\nTrain LSTM ...")

    # Load features
    oFeatureTrain = VideoFeatures(sFeatureDir + "/train", 
        oRNN.nFramesNorm, oRNN.nFeatureLength, oRNN.oClasses.liClasses)
    oFeatureVal = VideoFeatures(sFeatureDir + "/val", 
        oRNN.nFramesNorm, oRNN.nFeatureLength, oRNN.oClasses.liClasses)

    # Helper: Save results
    csv_logger = CSVLogger(sLogPath.split(".")[0] + "-acc.csv")

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    checkpointer = ModelCheckpoint(
        filepath = sModelDir + "/" + (sLogPath.split("/")[-1]).split(".")[0] + "-best.h5",
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
    sModelSaved = sModelDir + "/" + (sLogPath.split("/")[-1]).split(".")[0] + "-last.h5"
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
    sClassFile      = "data-set/04-chalearn/class.csv"
    #sVideoDir      = "data-set/04-chalearn"
    sFrameDir       = "data-temp/04-chalearn/%03d-oflow-20"%(nClasses)
    sFeatureDir     = "data-temp/04-chalearn/%03d-oflow-20-mobilenet"%(nClasses)
    sModelDir       = "model"
    sLogDir         = "log"

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
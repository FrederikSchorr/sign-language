"""
Classify human motion videos from ChaLearn dataset

ChaLearn dataset:
http://chalearnlap.cvc.uab.es/dataset/21/description/

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""

import os
import glob
import time
import sys

import numpy as np
import pandas as pd

import keras

sys.path.append(os.path.abspath("."))
from s7i3d.datagenerator import VideoClasses, FramesGenerator, FeaturesGenerator
from s7i3d.i3d_inception import Inception_Inflated3d, Inception_Inflated3d_Top


def train_i3d_top(sFeatureDir:str, sModelDir:str, sLogPath:str, keModel:keras.Model, oClasses: VideoClasses,
    nBatchSize:int=16, nEpoch:int=100, fLearn:float=1e-4):

    # Load training data
    genFeaturesVal   = FeaturesGenerator(sFeatureDir + "/val", nBatchSize,
        keModel.input_shape[1:], oClasses.liClasses)
    genFeaturesTrain = FeaturesGenerator(sFeatureDir + "/train", nBatchSize, 
        keModel.input_shape[1:], oClasses.liClasses)

    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger(sLogPath.split(".")[0] + "-acc.csv")

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath = sModelDir + "/" + (sLogPath.split("/")[-1]).split(".")[0] + "-best.h5",
        verbose = 1, save_best_only = True)

    # Use same optimizer as in https://github.com/deepmind/kinetics-i3d
    optimizer = keras.optimizers.SGD(lr = fLearn, momentum = 0.9, decay = 1e-7)
    keModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Fit!
    print("Fit with generator, learning rate %f ..." % fLearn)
    keModel.fit_generator(
        generator = genFeaturesTrain,
        validation_data = genFeaturesVal,
        epochs = nEpoch,
        workers = 0, #4,                 
        use_multiprocessing = False, #True,
        #max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, checkpointer])    
    
    # save model
    sModelSaved = sModelDir + "/" + (sLogPath.split("/")[-1]).split(".")[0] + "-last.h5"
    keModel.save(sModelSaved)

    return

def main():
   
    nClasses = 3

    # directories
    sClassFile       = "data-set/04-chalearn/class.csv"
    #sVideoDir       = "data-set/04-chalearn"
    #sFrameDir        = "data-temp/04-chalearn/%03d/frame"%(nClasses)
    sFrameFeatureDir = "data-temp/04-chalearn/%03d/frame-i3d"%(nClasses)
    #sFlowDir         = "data-temp/04-chalearn/%03d/oflow"%(nClasses)
    sFlowFeatureDir  = "data-temp/04-chalearn/%03d/oflow-i3d"%(nClasses)
    sModelDir        = "model"
    
 
    #sModelSaved = sModelDir + "/20180612-0740-lstm-13in249-last.h5"

    LEARNING_RATE = 1e-3
    EPOCHS = 100
    BATCHSIZE = 16

    print("\nStarting ChaLearn training in directory:", os.getcwd())

    # read the ChaLearn classes
    oClasses = VideoClasses(sClassFile)

    # RGB: Load empty i3d top layer and train it
    print("Load new I3D rgb top model ...")
    sLogPath = "log/" + time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-chalearn%03d-rgb-i3dtop.csv"%(nClasses)

    keI3D_top_rgb = Inception_Inflated3d_Top(oClasses.nClasses, dropout_prob=0.5)
    train_i3d_top(sFrameFeatureDir, sModelDir, sLogPath, keI3D_top_rgb, oClasses,
        BATCHSIZE, EPOCHS, LEARNING_RATE)

    # FLOW: Load empty i3d top layer and train it
    """print("Load new I3D flow top model ...")
    sLogPath = "log/" + time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-chalearn%03d-oflow-i3dtop.csv"%(nClasses)

    keI3D_top_flow = Inception_Inflated3d_Top(oClasses.nClasses, dropout_prob=0.5)
    train_i3d_top(sFlowFeatureDir, sModelDir, sLogPath, keI3D_top_flow, oClasses,
        BATCHSIZE, EPOCHS, LEARNING_RATE) """           

    return
    
    
if __name__ == '__main__':
    main()
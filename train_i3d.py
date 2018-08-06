"""
https://github.com/FrederikSchorr/sign-language

Train a pre-trained I3D convolutional network to classify videos
"""

import os
import glob
import time
import sys

import numpy as np
import pandas as pd

import keras
from keras import backend as K

from datagenerator import VideoClasses, FramesGenerator
from model_i3d import Inception_Inflated3d, add_i3d_top


def layers_freeze(keModel:keras.Model) -> keras.Model:
    
    print("Freeze all %d layers in Model %s" % (len(keModel.layers), keModel.name))
    for layer in keModel.layers:
        layer.trainable = False

    return keModel

def layers_unfreeze(keModel:keras.Model) -> keras.Model:
    
    print("Unfreeze all %d layers in Model %s" % (len(keModel.layers), keModel.name))
    for layer in keModel.layers:
        layer.trainable = True

    return keModel


def count_params(keModel:keras.Model):

    trainable_count = int(
        np.sum([K.count_params(p) for p in set(keModel.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(keModel.non_trainable_weights)]))

    print('Total params: {:,}'.format(trainable_count + non_trainable_count))
    print('Trainable params: {:,}'.format(trainable_count))
    print('Non-trainable params: {:,}'.format(non_trainable_count))
    
    return



def train_I3D_oflow_end2end(diVideoSet):
    """ 
    * Loads pretrained I3D model, 
    * reads optical flow data generated from training videos,
    * adjusts top-layers adequately for video data,
    * trains only news top-layers,
    * then fine-tunes entire neural network,
    * saves logs and models to disc.
    """
   
    # directories
    sFolder = "%03d-%d"%(diVideoSet["nClasses"], diVideoSet["nFramesNorm"])
    sClassFile       = "data-set/%s/%03d/class.csv"%(diVideoSet["sName"], diVideoSet["nClasses"])
    #sVideoDir        = "data-set/%s/%03d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    #sImageDir        = "data-temp/%s/%s/image"%(diVideoSet["sName"], sFolder)
    #sImageFeatureDir = "data-temp/%s/%s/image-i3d"%(diVideoSet["sName"], sFolder)
    sOflowDir        = "data-temp/%s/%s/oflow"%(diVideoSet["sName"], sFolder)
    #sOflowFeatureDir = "data-temp/%s/%s/oflow-i3d"%(diVideoSet["sName"], sFolder)
    
    sModelDir        = "model"

    diTrainTop = {
        "fLearn" : 1e-3,
        "nEpochs" : 3}

    diTrainAll = {
        "fLearn" : 1e-4,
        "nEpochs" : 17}

    nBatchSize = 4

    print("\nStarting I3D end2end training ...")
    print(os.getcwd())

    # read the ChaLearn classes
    oClasses = VideoClasses(sClassFile)

    # Load training data
    genFramesTrain = FramesGenerator(sOflowDir + "/train", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 2, oClasses.liClasses)
    genFramesVal = FramesGenerator(sOflowDir + "/val", nBatchSize, 
        diVideoSet["nFramesNorm"], 224, 224, 2, oClasses.liClasses)

    # Load pretrained i3d model and adjust top layer 
    print("Load pretrained I3D flow model ...")
    keI3DOflow = Inception_Inflated3d(
        include_top=False,
        weights='flow_imagenet_and_kinetics',
        input_shape=(diVideoSet["nFramesNorm"], 224, 224, 2))
    print("Add top layers with %d output classes ..." % oClasses.nClasses)
    keI3DOflow = layers_freeze(keI3DOflow)
    keI3DOflow = add_i3d_top(keI3DOflow, oClasses.nClasses, dropout_prob=0.5)
        
    # Prep logging
    sLog = time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-%s%03d-oflow-i3d"%(diVideoSet["sName"], diVideoSet["nClasses"])
    
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger("log/" + sLog + "-acc.csv", append = True)

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    cpTopLast = keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-above-last.h5", verbose = 0)
    cpTopBest = keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-above-best.h5",
        verbose = 1, save_best_only = True)
    cpAllLast = keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-last.h5", verbose = 0)
    cpAllBest = keras.callbacks.ModelCheckpoint(filepath = sModelDir + "/" + sLog + "-entire-best.h5",
        verbose = 1, save_best_only = True)

    # Fit top layers
    print("Fit I3D top layers with generator: %s" % (diTrainTop))
    optimizer = keras.optimizers.Adam(lr = diTrainTop["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)    

    keI3DOflow.fit_generator(
        generator = genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainTop["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpTopLast, cpTopBest])
    
    # Fit entire I3D model
    print("Finetune all I3D layers with generator: %s" % (diTrainAll))
    keI3DOflow = layers_unfreeze(keI3DOflow)
    optimizer = keras.optimizers.Adam(lr = diTrainAll["fLearn"])
    keI3DOflow.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    count_params(keI3DOflow)    

    keI3DOflow.fit_generator(
        generator = genFramesTrain,
        validation_data = genFramesVal,
        epochs = diTrainAll["nEpochs"],
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, cpAllLast, cpAllBest])

    return
    
    
if __name__ == '__main__':

    """diVideoSet = {"sName" : "ledasila",
        "nClasses" : 21,   # number of classes
        "nFramesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (288, 352), # height, width
        "nFpsAvg" : 25,
        "nFramesAvg" : 75,
        "fDurationAvg" : 3.0} # seconds
    """
    diVideoSet = {"sName" : "chalearn",
        "nClasses" : 20,   # number of classes
        "nFramesNorm" : 40,    # number of frames per video
        "nMinDim" : 240,   # smaller dimension of saved video-frames
        "tuShape" : (240, 320), # height, width
        "nFpsAvg" : 10,
        "nFramesAvg" : 50, 
        "fDurationAvG" : 5.0} # seconds 
    
    train_I3D_oflow_end2end(diVideoSet)
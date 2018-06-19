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
from s7i3d.preprocess import videos2frames
from s7i3d.datagenerator import VideoClasses, FramesGenerator
from s7i3d.i3d_inception import Inception_Inflated3d, add_top_layer


def layers_freeze(keModel:keras.Model) -> keras.Model:
    print("Freeze all layers ...")
    for layer in keModel.layers:
        layer.trainable = False

    return keModel


def train():

    return

def main():
   
    nClasses = 3

    # directories
    sClassFile       = "data-set/04-chalearn/class.csv"
    #sVideoDir       = "data-set/04-chalearn"
    sFrameDir        = "data-temp/04-chalearn/%03d-frame"%(nClasses)
    sFrameFeatureDir = "data-temp/04-chalearn/%03d-frame-i3d"%(nClasses)
    sFlowDir         = "data-temp/04-chalearn/%03d-oflow"%(nClasses)
    sFlowFeatureDir  = "data-temp/04-chalearn/%03d-oflow-i3d"%(nClasses)

    sModelDir       = "model"
    sLogPath        = "log/" + time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-chalearn%03d-oflow-i3dtop.csv"%(nClasses)
 
    #sModelSaved = sModelDir + "/20180612-0740-lstm-13in249-last.h5"

    NUM_FRAMES = 79
    FRAME_HEIGHT = 224
    FRAME_WIDTH = 224
    NUM_RGB_CHANNELS = 3
    NUM_FLOW_CHANNELS = 2

    LEARNING_RATE = 1e-1
    EPOCHS = 1
    BATCHSIZE = 4

    print("\nStarting ChaLearn training in directory:", os.getcwd())

    # read the ChaLearn classes
    oClasses = VideoClasses(sClassFile)

    # extract images
    #videos2frames(sVideoDir, sFrameDir, nClasses = None)

    # Load training data
    genFramesTrain = FramesGenerator(sFrameDir + "/train", BATCHSIZE, 
        NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS, oClasses.liClasses)
    genFramesVal = FramesGenerator(sFrameDir + "/val", BATCHSIZE,
        NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS, oClasses.liClasses)

    # Load pretrained i3d model and adjust top layer 
    print("Load pretrained I3D model ...")
    keI3D_rgb = Inception_Inflated3d(
        include_top=False,
        weights='rgb_imagenet_and_kinetics',
        input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS))
    print("Add top layers with %d output classes ..." % oClasses.nClasses)
    keI3D_rgb = layers_freeze(keI3D_rgb)
    keI3D_rgb = add_top_layer(keI3D_rgb, oClasses.nClasses, dropout_prob=0.5)

    # Use same optimizer as in https://github.com/deepmind/kinetics-i3d
    optimizer = keras.optimizers.SGD(lr = LEARNING_RATE, momentum = 0.9, decay = 1e-7)
    keI3D_rgb.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    #keI3D_rgb.summary()    
        
    # Prep logging
    os.makedirs(sLogDir, exist_ok=True)
    sLog = time.strftime("%Y%m%d-%H%M") + "-lstm-" + \
        str(genFramesTrain.nSamples + genFramesVal.nSamples) + "in" + str(oClasses.nClasses)
    
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger(sLogDir + "/" + sLog + "-acc.csv")

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath = sModelDir + "/" + sLog + "-best.h5",
        verbose = 1, save_best_only = True)
 
    # Fit!
    print("Fit with generator, learning rate %f ..." % LEARNING_RATE)
    keI3D_rgb.fit_generator(
        generator = genFramesTrain,
        validation_data = genFramesVal,
        epochs = EPOCHS,
        workers = 4,                 
        use_multiprocessing = True,
        max_queue_size = 8, 
        verbose = 1,
        callbacks=[csv_logger, checkpointer]
    )    
    
    # save model
    sModelSaved = sModelDir + "/" + sLog + "-last.h5"
    keI3D_rgb.save(sModelSaved)

    return
    
    
if __name__ == '__main__':
    main()
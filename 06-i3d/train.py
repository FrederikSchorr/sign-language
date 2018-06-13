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

import numpy as np
import pandas as pd

import category_encoders.one_hot

from video2frame import file2frame, frames_trim, image_crop, frames_show
from preprocess import videos2frames

from i3d_inception import Inception_Inflated3d, add_top_layer

import keras
#from keras.optimizers import SGD
#from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger


class VideoClasses():

    def __init__(self, sClassFile:str):
        # load label description: index, sClass, sLong, sCat, sDetail
        self.dfClass = pd.read_csv(sClassFile)
        self.liClasses = list(self.dfClass.sClass)
        self.nClasses = len(self.dfClass)
        return


class VideoFrames():
    """ Read (video) frames from disc, check shape, resize, onehot labels 
    
    Assume frames directories:
    ... / sPath / class / videoname / frame.jpg
    """

    def __init__(self, sPath:str, nFrames:int, nHeight:int, nWidth:int, nDepth:int, liClasses:list = None):
        
        # retrieve all videos = frame directories
        liPaths = glob.glob(sPath + "/*/*")
        self.nSamples = len(liPaths)
        if self.nSamples == 0: raise ValueError("Found no frame directories files in " + sPath)
        print("Load %d samples from %s ..." % (self.nSamples, sPath))

        self.arFrames = np.zeros((self.nSamples, nFrames, nHeight, nWidth, nDepth))
        self.liLabels = [] 
        
        # loop through all videos = frame directories
        for i in range(self.nSamples):
            
            # Get the frames from disc
            sFrameDir = liPaths[i]
            arFrames = file2frame(sFrameDir)
            
            # crop to centered image
            arFrames = image_crop(arFrames, nHeight, nWidth)
            
            # normalize the number of frames (if upsampling necessary => in the end)
            arFrames = frames_trim(arFrames, nFrames)
            
            if i % 100 == 0: print("  #%5d Video %s: frames %s" % (i, sFrameDir, str(arFrames.shape)))
            #frames_show(arFrames, 10)

            self.arFrames[i] = arFrames
            
            # Extract the label from path
            sLabel = sFrameDir.split("/")[-2]
            self.liLabels.append(sLabel)
            
        # extract unique classes from all detected labels
        self.liClasses = sorted(np.unique(self.liLabels))

        # if classes are provided upfront
        if liClasses != None:
            liClasses = sorted(np.unique(liClasses))
            # check detected vs provided classes
            if set(self.liClasses).issubset(set(liClasses)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            self.liClasses = liClasses
            
        self.nClasses = len(self.liClasses)
        
        # one hot encode labels, using above classes
        one_hot = category_encoders.one_hot.OneHotEncoder(
            return_df=False, handle_unknown="error", verbose = 1)
        one_hot.fit(self.liClasses)
        self.arLabelsOneHot = one_hot.transform(self.liLabels)
        
        print("Loaded %d samples from %d classes, labels onehot shape %s" % \
            (self.nSamples, self.nClasses, str(self.arLabelsOneHot.shape)))
        
        return
        

def layers_freeze(keModel:keras.Model) -> keras.Model:
    print("Freeze all layers ...")
    for layer in keModel.layers:
        layer.trainable = False

    return keModel


def main():
   
    # directories
    sClassFile = "../datasets/04-chalearn/class.csv"
    sVideoDir = "../datasets/04-chalearn"
    sFrameDir = "06-i3d/data/frame"

    sModelDir = "06-i3d/model"
    sLogDir = "06-i3d/log"

    #sModelSaved = sModelDir + "/20180612-0740-lstm-13in249-last.h5"

    NUM_FRAMES = 79
    FRAME_HEIGHT = 224
    FRAME_WIDTH = 224
    NUM_RGB_CHANNELS = 3
    NUM_FLOW_CHANNELS = 2

    LEARNING_RATE = 1e-2
    EPOCHS = 10
    BATCHSIZE = 4

    print("\nStarting ChaLearn training in directory:", os.getcwd())

    # read the ChaLearn classes
    oClasses = VideoClasses(sClassFile)

    # extract images
    videos2frames(sVideoDir, sFrameDir, nClasses = 20)

    # Load training data
    oFramesTrain = VideoFrames(sFrameDir + "/train", 
        NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS, oClasses.liClasses)
    oFramesVal = VideoFrames(sFrameDir + "/val", 
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
    keI3D_rgb.summary()    
        
    # prep logging
    os.makedirs(sLogDir, exist_ok=True)
    sLog = time.strftime("%Y%m%d-%H%M") + "-lstm-" + \
        str(oFramesTrain.nSamples + oFramesVal.nSamples) + "in" + str(oClasses.nClasses)
    
    # Helper: Save results
    csv_logger = keras.callbacks.CSVLogger(sLogDir + "/" + sLog + "-acc.csv")

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath = sModelDir + "/" + sLog + "-best.h5",
        verbose = 1, save_best_only = True)
 
    # Fit!
    keI3D_rgb.fit(
        oFramesTrain.arFrames,
        oFramesTrain.arLabelsOneHot,
        batch_size = BATCHSIZE,
        epochs = EPOCHS,
        verbose = 1,
        shuffle=True,
        validation_data=(oFramesVal.arFrames, oFramesVal.arLabelsOneHot),
        callbacks=[csv_logger, checkpointer]
    )    
    
    # save model
    sModelSaved = sModelDir + "/" + sLog + "-last.h5"
    keI3D_rgb.save(sModelSaved)

    return
    
    
if __name__ == '__main__':
    main()
"""
Use CNN-LSTM classifier model trained on ChaLearn dataset of human gestures
for transfer learning on smaller LedaSila dataset on Austrian sign language

ChaLearn dataset on human gestures:
http://chalearnlap.cvc.uab.es/dataset/21/description/

LedaSila dataset on Austrian sign language gestures:
http://ledasila.aau.at
by Alpen Adria Universität Klagenfurt

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""

import os
import glob
import sys
import time

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger


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


def transferlearning(sFeatureDir, sPretrainedModel, sModelDir, sLogDir, 
    nFramesNorm = 20, nFeatureLength = 2048, nBatchSize=16, nEpoch=100, fLearn=0.0001):

    
    # Load features
    oFeatureTrain = Features(sFeatureDir + "/train", nFramesNorm, nFeatureLength)
    oFeatureVal = Features(sFeatureDir + "/val", nFramesNorm, nFeatureLength)
    assert(oFeatureTrain.liLabels == oFeatureVal.liLabels)
    
    # load pretrained model and adjust last layers
    print("Loading pretrained LSTM neural network %s ..." % sPretrainedModel)
    keModel = load_model(sPretrainedModel)
    assert(keModel.input_shape == (None, nFramesNorm, nFeatureLength))
    
    # freeze  LSTM layer
    #keModel.get_layer("lstm_1").trainable = False
    keModel.get_layer("lstm_2").trainable = False
    
    # replace last  layer(s)
    keModel.pop() # last dense layer
    #keModel.pop() # dropout layer
    #keModel.pop() # dense layer after LSTM

    #keModel.add(Dense(256, activation = "relu"))
    #keModel.add(Dropout(0.5))
    keModel.add(Dense(oFeatureTrain.nLabels, activation = "softmax", name="dense_last"))
    
    keModel.summary()

    # Now compile the network.
    optimizer = Adam(fLearn)#, decay=1e-6)
    keModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Helper: Save results
    os.makedirs(sLogDir, exist_ok=True)
    sLog = time.strftime("%Y%m%d-%H%M") + "-transfer-" + \
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
    
    # save last model
    keModel.save(sModelDir + "/" + sLog + "-last.h5")

    return


def main():
   
    # directories
    sFeatureDir = "02b-ledasila/data/0440in21/feature"
    sPretrainedModel = "03a-chalearn/model/20180525-1033-lstm-35878in249-best.h5"
    sModelDir = "03b-transfer/model"
    sLogDir = "03b-transfer/log"

    nFramesNorm = 20 # number of frames per video for feature calculation
    nFeatureLength = 2048 # output features from CNN

    transferlearning(sFeatureDir, sPretrainedModel, sModelDir, sLogDir, 
                     nFramesNorm, nFeatureLength, nBatchSize=16, nEpoch=1000, fLearn=1e-5)
    return


if __name__ == '__main__':
    main()
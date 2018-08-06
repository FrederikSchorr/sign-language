"""
https://github.com/FrederikSchorr/sign-language

Define or load a Keras LSTM model.
"""

import os
import glob
import time
import sys
import warnings

import numpy as np
import pandas as pd

import keras


def lstm_build(nFramesNorm:int, nFeatureLength:int, nClasses:int, fDropout:float = 0.5) -> keras.Model:

    # Build new LSTM model
    print("Build and compile LSTM model ...")
    keModel = keras.models.Sequential()
    keModel.add(keras.layers.LSTM(nFeatureLength * 1, return_sequences=True,
        input_shape=(nFramesNorm, nFeatureLength),
        dropout=fDropout))
    keModel.add(keras.layers.LSTM(nFeatureLength * 1, return_sequences=False, dropout=fDropout))
    keModel.add(keras.layers.Dense(nClasses, activation='softmax'))

    keModel.summary()

    return keModel


def lstm_load(sPath:str, nFramesNorm:int, nFeatureLength:int, nClasses:int) -> keras.Model:

    print("Load trained LSTM model from %s ..." % sPath)
    keModel = keras.models.load_model(sPath)
    
    tuInputShape = keModel.input_shape[1:]
    tuOutputShape = keModel.output_shape[1:]
    print("Loaded input shape %s, output shape %s" % (str(tuInputShape), str(tuOutputShape)))

    if tuInputShape != (nFramesNorm, nFeatureLength):
        raise ValueError("Unexpected LSTM input shape")
    if tuOutputShape != (nClasses, ):
        raise ValueError("Unexpected LSTM output shape")

    return keModel
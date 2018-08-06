"""
https://github.com/FrederikSchorr/sign-language

Load pretrained MobileNet or Inception CNN and perform some checks.
Models are loaded without top layers, suitable for feature extraction.
"""

import os
import glob
import time
import sys
import warnings

import numpy as np
import pandas as pd

import keras


def features_2D_load_model(diFeature:dict) -> keras.Model:
    sModelName = diFeature["sName"]
    print("Load 2D extraction model %s ..." % sModelName)

    # load pretrained keras models
    if sModelName == "mobilenet":

        # load base model with top
        keBaseModel = keras.applications.mobilenet.MobileNet(
            weights="imagenet",
            input_shape = (224, 224, 3),
            include_top = True)

        # We'll extract features at the final pool layer
        keModel = keras.models.Model(
            inputs=keBaseModel.input,
            outputs=keBaseModel.get_layer('global_average_pooling2d_1').output,
            name=sModelName + " without top layer") 

    elif sModelName == "inception":

        # load base model with top
        keBaseModel = keras.applications.inception_v3.InceptionV3(
            weights='imagenet',
            include_top=True)

        # We'll extract features at the final pool layer
        keModel = keras.models.Model(
            inputs=keBaseModel.input,
            outputs=keBaseModel.get_layer('avg_pool').output,
            name=sModelName + " without top layer") 
    
    else: raise ValueError("Unknown 2D feature extraction model")

    # check input & output dimensions
    tuInputShape = keModel.input_shape[1:]
    tuOutputShape = keModel.output_shape[1:]
    print("Expected input shape %s, output shape %s" % (str(tuInputShape), str(tuOutputShape)))

    if tuInputShape != diFeature["tuInputShape"]:
        raise ValueError("Unexpected input shape")
    if tuOutputShape != diFeature["tuOutputShape"]:
        raise ValueError("Unexpected output shape")

    return keModel
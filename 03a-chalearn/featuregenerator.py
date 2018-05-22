"""
Class for managing our data.

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""
import csv
import numpy as np
import pandas as pd

import random

import glob
import os.path
import sys

import operator
import threading

from keras.utils import to_categorical

class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)

def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen

class FeatureGenerator():

    def __init__(self, sPath, sTrain, sTest):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """

        self.sPath = sPath

        # Get the data.
        self.dfTrain = self.get_data(os.path.join(sPath, sTrain))
        self.nTrain = len(self.dfTrain)

        self.dfTest = self.get_data(os.path.join(sPath, sTest))
        self.nTest = len(self.dfTest)

        # Get the classes.
        self.liClasses = list(np.sort(self.dfTrain.sClass.unique()))
        self.nClasses = len(self.liClasses)
        print("%d train samples + %d test samples in %d classes" % \
            (self.nTrain, self.nTest, self.nClasses))


    @staticmethod
    def get_data(sPathData):
        """Load filenames in all subdirectories"""
        
        dfFiles = pd.DataFrame(glob.glob(os.path.join(sPathData, "*", "*.npy")), 
            dtype=str, columns=["sPath"])

        # extract class names
        dfFiles["sClass"] = dfFiles.sPath.apply(lambda s: s.split("/")[-2])

        return dfFiles
           

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.liClasses.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, self.nClasses)

        assert len(label_hot) == self.nClasses

        return label_hot


    @threadsafe_generator
    def frame_generator(self, sTrain_test, batch_size, nFramesNorm, nFeatureLength):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:
        """
        # Get the right dataset for the generator.
        if sTrain_test == "train": dfFiles = self.dfTrain
        elif sTrain_test == "val": dfFiles = self.dfTest
        else: raise ValueError("Specify train oder validation dataset")

        print("Creating <%s> generator with %d samples in %d classes" % \
            (sTrain_test, dfFiles.shape[0], self.nClasses))

        while 1:
            X, y = [], []

            # Generate batch_size samples - think this to random - need to improve
            for _, seFile in dfFiles.sample(batch_size).iterrows():

                # Get the sequence from disk.
                arFeatures = np.load(str(seFile.sPath))
                if (arFeatures.shape != (nFramesNorm, nFeatureLength)):
                    print("Error: %s has wrong shape %s" % (seFile.sPath, arFeatures.shape))
                
                X.append(arFeatures)
                y.append(self.get_class_one_hot(str(seFile.sClass)))

            yield np.array(X), np.array(y)

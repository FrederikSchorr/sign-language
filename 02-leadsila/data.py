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

class DataSet():

    def __init__(self, sPath):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """

        self.sPath = sPath

        # Get the data.
        self.dfTrain = self.get_data(os.path.join(sPath, "train"))
        self.dfTest = self.get_data(os.path.join(sPath, "test"))

        # Get the classes.
        self.a1Classes = np.sort(self.dfTrain.sWord.unique())
        self.nClasses = len(self.a1Classes)


    @staticmethod
    def get_data(sPath):
        """Load filenames in all subdirectories"""
        dfFiles = pd.DataFrame(glob.glob(os.path.join(sPath, "**"), recursive=True), \
            dtype=str, columns=["sPath"])

        # extract file names
        dfFiles["sFile"] = dfFiles.sPath.apply(lambda s: s.split("/")[-1])

        # drop files/folders without extension
        dfFiles.loc[dfFiles.sFile.apply(lambda s: len(s.split("."))<=1),:] = None
        dfFiles.dropna(inplace=True)

        # extract word
        dfFiles["sWord"] = dfFiles.sFile.apply(lambda s: s.split("-")[0])

        return dfFiles
           

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.a1Classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.a1Classes))

        assert len(label_hot) == len(self.a1Classes)

        return label_hot


    @threadsafe_generator
    def frame_generator(self, sTrain_test="train", batch_size=32):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:
        """
        # Get the right dataset for the generator.
        if sTrain_test == "train": dfFiles = self.dfTrain
        elif sTrain_test == "test": dfFiles = self.dfTest
        else: raise ValueError("Specify train oder test dataset")

        print("Creating %s generator with %d samples in %d categories" % \
            (sTrain_test, dfFiles.shape[0], len(self.a1Classes)))

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(batch_size):

                # Get a random sample.
                seFile = random.choice(dfFiles)

                # Get the sequence from disk.
                liFeatures = np.load(seFile.sPath)

                if liFeatures is None:
                    raise ValueError("Can't find features. Did you generate them?")

                X.append(liFeatures)
                y.append(self.get_class_one_hot(seFile.sWord))

            yield np.array(X), np.array(y)

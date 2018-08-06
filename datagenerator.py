"""
https://github.com/FrederikSchorr/sign-language

For neural network training the method Keras.model.fit_generator is used. 
This requires a generator that reads and yields training data to the Keras engine.
"""


import glob
import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import keras

from frame import files2frames, images_normalize, frames_show

class FramesGenerator(keras.utils.Sequence):
    """Read and yields video frames/optical flow for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, sPath:str, \
        nBatchSize:int, nFrames:int, nHeight:int, nWidth:int, nChannels:int, \
        liClassesFull:list = None, bShuffle:bool = True):
        """
        Assume directory structure:
        ... / sPath / class / videoname / frames.jpg
        """

        'Initialization'
        self.nBatchSize = nBatchSize
        self.nFrames = nFrames
        self.nHeight = nHeight
        self.nWidth = nWidth
        self.nChannels = nChannels
        self.tuXshape = (nFrames, nHeight, nWidth, nChannels)
        self.bShuffle = bShuffle

        # retrieve all videos = frame directories
        self.dfVideos = pd.DataFrame(sorted(glob.glob(sPath + "/*/*")), columns=["sFrameDir"])
        self.nSamples = len(self.dfVideos)
        if self.nSamples == 0: raise ValueError("Found no frame directories files in " + sPath)
        print("Detected %d samples in %s ..." % (self.nSamples, sPath))

        # extract (text) labels from path
        seLabels =  self.dfVideos.sFrameDir.apply(lambda s: s.split("/")[-2])
        self.dfVideos.loc[:, "sLabel"] = seLabels
            
        # extract unique classes from all detected labels
        self.liClasses = sorted(list(self.dfVideos.sLabel.unique()))

        # if classes are provided upfront
        if liClassesFull != None:
            liClassesFull = sorted(np.unique(liClassesFull))
            # check detected vs provided classes
            if set(self.liClasses).issubset(set(liClassesFull)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            self.liClasses = liClassesFull
            
        self.nClasses = len(self.liClasses)

        # encode labels
        trLabelEncoder = LabelEncoder()
        trLabelEncoder.fit(self.liClasses)
        self.dfVideos.loc[:, "nLabel"] = trLabelEncoder.transform(self.dfVideos.sLabel)
        
        self.on_epoch_end()
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nSamples / self.nBatchSize))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples)
        if self.bShuffle == True:
            np.random.shuffle(self.indexes)


    def __getitem__(self, nStep):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.nBatchSize:(nStep+1)*self.nBatchSize]

        # get batch of videos
        dfVideosBatch = self.dfVideos.loc[indexes, :]
        nBatchSize = len(dfVideosBatch)

        # initialize arrays
        arX = np.empty((nBatchSize, ) + self.tuXshape, dtype = float)
        arY = np.empty((nBatchSize), dtype = int)

        # Generate data
        for i in range(nBatchSize):
            # generate data for single video(frames)
            arX[i,], arY[i] = self.__data_generation(dfVideosBatch.iloc[i,:])
            #print("Sample #%d" % (indexes[i]))

        # onehot the labels
        return arX, keras.utils.to_categorical(arY, num_classes=self.nClasses)


    def __data_generation(self, seVideo:pd.Series) -> (np.array(float), int):
        "Returns frames for 1 video, including normalizing & preprocessing"
       
        # Get the frames from disc
        ar_nFrames = files2frames(seVideo.sFrameDir)

        # only use the first nChannels (typically 3, but maybe 2 for optical flow)
        ar_nFrames = ar_nFrames[..., 0:self.nChannels]
        
        ar_fFrames = images_normalize(ar_nFrames, self.nFrames, self.nHeight, self.nWidth, bRescale = True)
        
        return ar_fFrames, seVideo.nLabel

    def data_generation(self, seVideo:pd.Series) -> (np.array(float), int):
        return self.__data_generation(seVideo)


class FeaturesGenerator(keras.utils.Sequence):
    """Reads and yields (preprocessed) I3D features for Keras.model.fit_generator
    Generator can be used for multi-threading.
    Substantial initialization and checks upfront, including one-hot-encoding of labels.
    """

    def __init__(self, sPath:str, nBatchSize:int, tuXshape, \
        liClassesFull:list = None, bShuffle:bool = True):
        """
        Assume directory structure:
        ... / sPath / class / feature.npy
        """

        'Initialization'
        self.nBatchSize = nBatchSize
        self.tuXshape = tuXshape
        self.bShuffle = bShuffle

        # retrieve all feature files
        self.dfSamples = pd.DataFrame(sorted(glob.glob(sPath + "/*/*.npy")), columns=["sPath"])
        self.nSamples = len(self.dfSamples)
        if self.nSamples == 0: raise ValueError("Found no feature files in " + sPath)
        print("Detected %d samples in %s ..." % (self.nSamples, sPath))

        # test shape of first sample
        arX = np.load(self.dfSamples.sPath[0])
        if arX.shape != tuXshape: raise ValueError("Wrong feature shape: " + str(arX.shape) + str(tuXshape))

        # extract (text) labels from path
        seLabels =  self.dfSamples.sPath.apply(lambda s: s.split("/")[-2])
        self.dfSamples.loc[:, "sLabel"] = seLabels
            
        # extract unique classes from all detected labels
        self.liClasses = sorted(list(self.dfSamples.sLabel.unique()))

        # if classes are provided upfront
        if liClassesFull != None:
            liClassesFull = sorted(np.unique(liClassesFull))
            # check detected vs provided classes
            if set(self.liClasses).issubset(set(liClassesFull)) == False:
                raise ValueError("Detected classes are NOT subset of provided classes")
            # use superset of provided classes
            self.liClasses = liClassesFull
            
        self.nClasses = len(self.liClasses)

        # encode labels
        trLabelEncoder = LabelEncoder()
        trLabelEncoder.fit(self.liClasses)
        self.dfSamples.loc[:, "nLabel"] = trLabelEncoder.transform(self.dfSamples.sLabel)
        
        self.on_epoch_end()
        return

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.nSamples / self.nBatchSize))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples)
        if self.bShuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, nStep):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.nBatchSize:(nStep+1)*self.nBatchSize]

        # Find selected samples
        dfSamplesBatch = self.dfSamples.loc[indexes, :]
        nBatchSize = len(dfSamplesBatch)

        # initialize arrays
        arX = np.empty((nBatchSize, ) + self.tuXshape, dtype = float)
        arY = np.empty((nBatchSize), dtype = int)

        # Generate data
        for i in range(nBatchSize):
            # generate single sample data
            arX[i,], arY[i] = self.__data_generation(dfSamplesBatch.iloc[i,:])

        # onehot the labels
        return arX, keras.utils.to_categorical(arY, num_classes=self.nClasses)

    def __data_generation(self, seSample:pd.Series) -> (np.array(float), int):
        'Generates data for 1 sample' 

        arX = np.load(seSample.sPath)

        return arX, seSample.nLabel



class VideoClasses():
    """
    Loads the video classes (incl descriptions) from a csv file
    """
    def __init__(self, sClassFile:str):
        # load label description: index, sClass, sLong, sCat, sDetail
        self.dfClass = pd.read_csv(sClassFile)

        # sort the classes
        self.dfClass = self.dfClass.sort_values("sClass").reset_index(drop=True)
        
        self.liClasses = list(self.dfClass.sClass)
        self.nClasses = len(self.dfClass)

        print("Loaded %d classes from %s" % (self.nClasses, sClassFile))
        return
import glob
import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
import keras

sys.path.append(os.path.abspath("."))
from s7i3d.video2frame import file2frames, frames_trim, image_crop, image_rescale, frames_show

class FramesGenerator(keras.utils.Sequence):
    """Read and yields video frames for Keras.model.fit_generator
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
        self.dfVideos = pd.DataFrame(glob.glob(sPath + "/*/*"), columns=["sFrameDir"])
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
        return int(np.floor(self.nSamples / self.nBatchSize))

    def __getitem__(self, nStep):
        'Generate one batch of data'

        # Generate indexes of the batch
        indexes = self.indexes[nStep*self.nBatchSize:(nStep+1)*self.nBatchSize]

        # Find list of IDs
        dfVideosBatch = self.dfVideos.loc[indexes, :]
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]
        nBatchSize = len(dfVideosBatch)

        # initialize arrays
        arX = np.empty((nBatchSize, ) + self.tuXshape, dtype = float)
        arY = np.empty((nBatchSize), dtype = int)

        # Generate data
        for i in range(nBatchSize):
            # generate data for single video(frames)
            arX[i,], arY[i] = self.__data_generation(dfVideosBatch.iloc[i,:])

        # onehot the labels
        return arX, keras.utils.to_categorical(arY, num_classes=self.nClasses)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.nSamples)
        if self.bShuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, seVideo:pd.Series) -> (np.array(float), int):
        'Generates data for 1 sample' 
       
        # Get the frames from disc
        ar_nFrames = file2frames(seVideo.sFrameDir)

        # only use the first nChannels (typically 3, but maybe 2 for optical flow)
        ar_nFrames = ar_nFrames[..., 0:self.nChannels]
        
        # crop to centered image
        ar_nFrames = image_crop(ar_nFrames, self.nHeight, self.nWidth)

        # normalize to [-1.0, 1.0]
        ar_fFrames = image_rescale(ar_nFrames)
        
        # normalize the number of frames (if upsampling necessary => in the end, otherwise better early)
        ar_fFrames = frames_trim(ar_fFrames, self.nFrames)

        return ar_fFrames, seVideo.nLabel


class VideoClasses():

    def __init__(self, sClassFile:str):
        # load label description: index, sClass, sLong, sCat, sDetail
        self.dfClass = pd.read_csv(sClassFile)
        self.liClasses = list(self.dfClass.sClass)
        self.nClasses = len(self.dfClass)
        return

"""

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""

import os
import glob

from PIL import Image

import numpy as np
import pandas as pd

import category_encoders.one_hot

import keras

VERBOSE = True


class VideoClasses():

    def __init__(self, sClassFile:str):
        # load label description: index, sClass, sLong, sCat, sDetail
        self.dfClass = pd.read_csv(sClassFile)
        self.liClasses = list(self.dfClass.sClass)
        self.nClasses = len(self.dfClass)
        return


class VideoFeatures():
    """ Read video2frame2features from disc, check size and save in arrays """

    def __init__(self, sPath:str, nFramesNorm:int, nFeatureLength:int, liClasses:list):
        
        # retrieve all feature files. Expected folder structure .../train/label/feature.npy
        self.liPaths = glob.glob(os.path.join(sPath, "*", "*.npy"))
        self.nSamples = len(self.liPaths)
        if self.nSamples == 0:
            print("Error: found no feature files in %s" % sPath)
        print("Load %d samples from %s ..." % (self.nSamples, sPath))

        self.arFeatures = np.zeros((self.nSamples, nFramesNorm, nFeatureLength))
        
        # loop through all feature files
        self.liLabels = [] 
        for i in range(self.nSamples):
            
            # Get the sequence from disc
            sFile = self.liPaths[i]
            arF = np.load(sFile)
            if (arF.shape != (nFramesNorm, nFeatureLength)):
                print("ERROR: %s has wrong shape %s" % (sFile, arF.shape))
            self.arFeatures[i] = arF
            
            # Extract the label from path
            sLabel = sFile.split("/")[-2]
            self.liLabels.append(sLabel)
            
        # extract classes from samples
        self.liClasses = sorted(np.unique(self.liLabels))
        # check if classes are already provided
        if liClasses != None:
            liClasses = sorted(np.unique(liClasses))
            # check detected vs provided classes
            if set(self.liClasses).issubset(set(liClasses)) == False:
                # detected classes are NOT subset of provided classes
                print("ERROR: unexpected additional classes detected")
                print("Provided classes:", liClasses)
                print("Detected classes:", self.liClasses)
                assert False
            # use superset of provided classes
            self.liClasses = liClasses
            
        self.nClasses = len(self.liClasses)
        
        # one hot encode labels, using above classes
        one_hot = category_encoders.one_hot.OneHotEncoder(
            return_df=False, handle_unknown="error", verbose = VERBOSE)
        one_hot.fit(self.liClasses)
        self.arLabelsOneHot = one_hot.transform(self.liLabels)
        #print("OneHot shape", self.arLabelsOneHot.shape, self.liClasses)
        
        print("Loaded %d samples from %d classes" % (self.nSamples, self.nClasses))
        
        return


class ConvNet():
    def __init__(self, sName):
        self.sName = sName

        if sName == "mobilenet":
            self.tuHeightWidth = (224, 224)
            self.nOutputFeatures = 1024
        elif sName == "inception":
            self.tuHeightWidth = (299, 299)
            self.nOutputFeatures = 2048
        else: raise ValueError("Unknown CNN model")

        return

    def load_model(self):
        # load pretrained keras models
        if self.sName == "mobilenet":
            self.preprocess_input = keras.applications.mobilenet.preprocess_input
            keBaseModel = keras.applications.mobilenet.MobileNet(
                weights="imagenet",
                input_shape = self.tuHeightWidth + (3,),
                include_top = True
            )
            # We'll extract features at the final pool layer
            self.keModel = keras.models.Model(
                inputs=keBaseModel.input,
                outputs=keBaseModel.get_layer('global_average_pooling2d_1').output
            ) 

        elif self.sName == "inception":
            self.tuHeightWidth = (299, 299)
            self.preprocess_input = keras.applications.inception_v3.preprocess_input
            keBaseModel = keras.applications.inception_v3.InceptionV3(
                weights='imagenet',
                include_top=True
            )
            # We'll extract features at the final pool layer
            self.keModel = keras.models.Model(
                inputs=keBaseModel.input,
                outputs=keBaseModel.get_layer('avg_pool').output
            )
        
        else: raise ValueError("Unknown CNN model")

        #self.keModel.summary()

        return

    def resize_transform(self, liImages):
        """ Resizing images to required size and transform to array of arrays

        # Parameter
            li_arImages ... list of images (as array)

        # Returns
            array of images (as array)
        """

        # make sure image as array
        if type(liImages[0]) is not np.ndarray: 
            raise ValueError("Expected array instead of", str(type(liImages[0])))

        li_ar = []
        for arImage in liImages:
            pilImage = keras.preprocessing.image.array_to_img(arImage)
            pilImage = pilImage.resize(self.tuHeightWidth)
            arImage = keras.preprocessing.image.img_to_array(pilImage)
            li_ar.append(arImage)

        return np.array(li_ar)


class RecurrentNet():

    def __init__(self, sName:str, nFramesNorm:int, nFeatureLength:int, oClasses:VideoClasses):
        self.sName = sName
        self.nFramesNorm = nFramesNorm
        self.nFeatureLength = nFeatureLength
        self.oClasses = oClasses
        print("Init RNN %s with %d number of frames, %d feature length, %d classes" % \
            (sName, nFramesNorm, nFeatureLength, oClasses.nClasses))
        return

    def build_compile(self, fLearn = 1e-4):
        if self.sName != "lstm": raise ValueError("Unexpected RNN")

        # Build new model
        print("Build and compile %s ..." % self.sName)
        self.keModel = keras.models.Sequential()
        self.keModel.add(keras.layers.LSTM(1024, return_sequences=True,
                        input_shape=(self.nFramesNorm, self.nFeatureLength),
                        dropout=0.5))
        self.keModel.add(keras.layers.LSTM(1024, return_sequences=False, dropout=0.5))
        #keModel.add(Dense(256, activation='relu'))
        #keModel.add(Dropout(0.5))
        self.keModel.add(keras.layers.Dense(self.oClasses.nClasses, activation='softmax'))

        # Now compile the network.
        optimizer = keras.optimizers.Adam(lr=fLearn)
        self.keModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        #self.keModel.summary()
        return    

    def load_model(self, sModelPath):
        print("Load neural network {} from {} ...".format(self.sName, sModelPath))
        self.keModel = keras.models.load_model(sModelPath)
        self.keModel.summary()
        
        if self.keModel.input_shape != (None, self.nFramesNorm, self.nFeatureLength):
            raise ValueError("Input shapes do not match")

        return
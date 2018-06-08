"""
Classify human motion videos from ChaLearn dataset

ChaLearn dataset:
http://chalearnlap.cvc.uab.es/dataset/21/description/

Code based on:
https://github.com/harvitronix/five-video-classification-methods
"""

import numpy as np
import pandas as pd

from math import ceil

import os
import glob
import shutil

from subprocess import call, check_output
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#from inceptionfeatures import InceptionV3_features

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from keras.applications import mobilenet
from keras.preprocessing import image


def video2frames(sVideoDir, sFrameDir, nFramesNorm = 20, nClasses = None):
    """ Extract frames from videos """
    
    # do not (partially) overwrite existing frame directory
    if os.path.exists(sFrameDir): raise ValueError("Folder {} alredy exists".format(sFrameDir)) 

    # get videos. Assume sVideoDir / train / class / video.avi
    dfVideos = pd.DataFrame(glob.glob(sVideoDir + "/*/*/*.avi"), columns=["sVideoPath"])
    print("Located {} videos in {}, extracting {} frames each to {} ..."\
        .format(len(dfVideos), sVideoDir, nFramesNorm, sFrameDir))

    # eventually restrict to first nLabels
    if nClasses != None:
        dfVideos.loc[:,"sLabel"] = dfVideos.sVideoPath.apply(lambda s: s.split("/")[-2])
        liClasses = sorted(dfVideos.sLabel.unique())[:nClasses]
        dfVideos = dfVideos[dfVideos["sLabel"].isin(liClasses)]
        print("Using only {} videos from {} classes".format(len(dfVideos), nClasses))

    nCounter = 0
    # loop through all videos and extract frames
    for sVideoPath in dfVideos.sVideoPath:
        
        # source path: ... / sVideoDir / train / class / video.avi
        li_sVideoPath = sVideoPath.split("/")
        if len(li_sVideoPath) < 4: raise ValueError("Video path should have min 4 components: {}".format(str(li_sVideoPath)))
        sVideoName = li_sVideoPath[-1].split(".")[0]

        # create frame directory for each video
        sDir = sFrameDir + "/" + li_sVideoPath[-3] + "/" + li_sVideoPath[-2] + "/" + sVideoName
        os.makedirs(sDir, exist_ok=True)
        
        # determine length of video in sec and deduce frame rate
        fVideoSec = int(check_output(["mediainfo", '--Inform=Video;%Duration%', sVideoPath]))/1000.0
        #nFramesPerSec = int(ceil(nFramesNorm / fVideoSec))
        fFramesPerSec = nFramesNorm / fVideoSec

        # call ffmpeg to extract frames from videos
        sFrames = sDir + "/frame-%03d.jpg"
        call(["ffmpeg", "-loglevel", "error" ,"-y", "-i", sVideoPath, \
            "-r", str(fFramesPerSec), "-frames", str(nFramesNorm), sFrames])
            
        # check right number of frames
        nFrames = len(glob.glob(sDir + "/*.jpg"))
        print("%5d | %s | %2.3fsec | %.1ffps | %df" % \
            (nCounter, sDir, fVideoSec, fFramesPerSec, nFrames))
        if nFrames != nFramesNorm: raise ValueError("Incorrect number of frames extracted")
        nCounter += 1

    # check number of created frames
    nFramesTotal = len(glob.glob(sFrameDir + "/*/*/*/*.jpg"))
    print("%d frames extracted from %d videos" % (nFramesTotal, len(dfVideos)))

    return


def frames2features(sConvNet, sFrameDir, sFeatureDir, nFramesNorm, nClasses=None):
    """ Use pretrained CNN to calculate features from video-frames """

    # do not (partially) overwrite existing feature directory
    if os.path.exists(sFeatureDir): raise ValueError("Folder {} alredy exists".format(sFeatureDir)) 

    sCurrentDir = os.getcwd()
    # get list of directories with frames: ... / sFrameDir/train/class/videodir/frames.jpg
    os.chdir(sFrameDir)
    dfVideos = pd.DataFrame(glob.glob("*/*/*"), dtype=str, columns=["sFrameDir"])
    os.chdir(sCurrentDir)
    print("Found %d directories=videos with frames" % len(dfVideos))

    # eventually restrict to first nLabels
    if nClasses != None:
        dfVideos.loc[:,"sLabel"] = dfVideos.sFrameDir.apply(lambda s: s.split("/")[-2])
        liClasses = sorted(dfVideos.sLabel.unique())[:nClasses]
        dfVideos = dfVideos[dfVideos["sLabel"].isin(liClasses)]
        print("Using only %d directories from %d classes" % (len(dfVideos), nClasses))

    # load ConvNet
    if sConvNet == "inception":
        # get the InceptionV3 model
        assert(False) # need to update code to work with inception model
        #keConvNet = InceptionV3_features()
    else:
        # default = MobileNet
        keConvNet = mobilenet.MobileNet(
            weights="imagenet",
            input_shape = (224, 224, 3),
            include_top = False
        )

    # loop over all videos-directories. 
    # Feed all frames into ConvNet, save results in file per video
    nCounter = 0
    for _, seVideo in dfVideos.iterrows():

        # save resulting list of features in file in eg "data/feature/train/label/video.npy"
        sFeaturePath = os.path.join(sFeatureDir, seVideo.sFrameDir + ".npy")
        if (os.path.exists(sFeaturePath)):
            # if feature already etracted, skip
            print("%5d | Features %s already exist" % (nCounter, sFeaturePath))
            continue
        os.makedirs(os.path.dirname(sFeaturePath), exist_ok=True)

        # retrieve frame files - in ascending order
        sFrames = sorted(glob.glob(os.path.join(sFrameDir, seVideo.sFrameDir, "*.jpg")))
        print("%5d | Extracting features from %d frames to %s" % (nCounter, len(sFrames), sFeaturePath))
        assert(len(sFrames) == nFramesNorm)

        # assemble array with all images
        liFrames = []
        for sFrame in sFrames:
            
            # load frame
            pilFrame = image.load_img(sFrame, target_size=(224, 224))
            arFrame = image.img_to_array(pilFrame)
            liFrames.append(arFrame)  
        
        # preprocessing - specific to ConvNet
        arFrames = mobilenet.preprocess_input(np.array(liFrames))
        
        # predict features with ConvNet
        arFeatures = keConvNet.predict(arFrames)
       
        # flatten the features
        arFeatures = arFeatures.reshape(nFramesNorm, -1)

        # save to file
        np.save(sFeaturePath, arFeatures)
        nCounter += 1
    return


class Features():
    """ Read features from disc, check size and save in arrays """

    def __init__(self, sPath, tuShape, liClasses = None):
        
        # retrieve all feature files. Expected folder structure .../train/label/feature.npy
        self.liPaths = glob.glob(os.path.join(sPath, "*", "*.npy"))
        self.nSamples = len(self.liPaths)
        if self.nSamples == 0:
            print("Error: found no feature files in %s" % sPath)
        print("Load %d samples from %s ..." % (self.nSamples, sPath))

        self.arFeatures = np.zeros((self.nSamples,) + tuShape)
        
        # loop through all feature files
        self.liLabels = [] 
        for i in range(self.nSamples):
            
            # Get the sequence from disc
            sFile = self.liPaths[i]
            arF = np.load(sFile)
            if (arF.shape != tuShape):
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
        label_encoder = LabelEncoder()
        label_encoder.fit(self.liClasses)
        ar_n_Labels = label_encoder.transform(self.liLabels).reshape(-1,1)
        onehot_encoder = OneHotEncoder(sparse=False)
        self.arLabelsOneHot = onehot_encoder.fit_transform(ar_n_Labels)
        
        print("Loaded %d samples from %d classes" % (self.nSamples, self.nClasses))
        
        return

        
def train(sConvNet, sFeatureDir, sModelSaved, sModelDir, sLogDir, 
          nFramesNorm = 20, nBatchSize=16, nEpoch=100, fLearn=1e-4):
    print("\nTrain LSTM ...")

    # output shape of ConvNet
    if sConvNet == "mobilenet":
        tuFeatureShape = (nFramesNorm, 50176)
    else: assert False

    # Load features
    oFeatureTrain = Features(sFeatureDir + "/train", tuFeatureShape)
    oFeatureVal = Features(sFeatureDir + "/val", tuFeatureShape, liClasses = oFeatureTrain.liClasses)

    # prep logging
    os.makedirs(sLogDir, exist_ok=True)
    sLog = time.strftime("%Y%m%d-%H%M") + "-lstm-" + \
        str(oFeatureTrain.nSamples + oFeatureVal.nSamples) + "in" + str(oFeatureTrain.nClasses)

    # save class names
    dfClass = pd.DataFrame(oFeatureTrain.liClasses, columns=["sClass"])
    sClassFile = sLogDir + "/" + sLog + "-class.csv"
    dfClass.to_csv(sClassFile)
    
    # Build a simple LSTM network. We pass the extracted features from
    # our CNN to this model     
    
    if sModelSaved == None:
        # Build new model
        keModel = Sequential()
        keModel.add(LSTM(1024, return_sequences=True,
                        input_shape=(20, 1024),
                        dropout=0.5))
        keModel.add(LSTM(1024, return_sequences=False, dropout=0.5))
        #keModel.add(Dense(256, activation='relu'))
        #keModel.add(Dropout(0.5))
        keModel.add(Dense(oFeatureTrain.nClasses, activation='softmax'))

        # Now compile the network.
        optimizer = Adam(lr=fLearn)
        keModel.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    else:
        # load model from file
        print("Loading saved LSTM neural network %s ..." % sModelSaved)
        keModel = load_model(sModelSaved)
        assert keModel.input_shape == ((None, ) + tuFeatureShape)

    keModel.summary()
    # Helper: TensorBoard
    #tb = TensorBoard(log_dir=os.path.join("../data/90-logs", model))

    # Helper: Stop when we stop learning.
    #early_stopper = EarlyStopping(patience=5)

    # Helper: Save results
    csv_logger = CSVLogger(os.path.join(sLogDir, sLog + '-acc.csv'))

    # Helper: Save the model
    os.makedirs(sModelDir, exist_ok=True)
    checkpointer = ModelCheckpoint(
        filepath = sModelDir + "/" + sLog + "-best.h5",
        verbose = 1, save_best_only = True)
 
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
        callbacks=[csv_logger, checkpointer]
    )    
    
    # save model
    sModelSaved = sModelDir + "/" + sLog + "-last.h5"
    keModel.save(sModelSaved)

    return sClassFile, sModelSaved


def evaluate(sFeatureDir, sModelSaved, nFramesNorm, nFeatureLength):    
    """ evaluate all features in given directory on saved model """
    print("\nEvaluate LSTM ...")

    # Load features
    oFeatures = Features(sFeatureDir, nFramesNorm, nFeatureLength)

    # load model from file
    print("Loading saved LSTM neural network %s ..." % sModelSaved)
    keModel = load_model(sModelSaved)
    assert(keModel.input_shape == (None, nFramesNorm, nFeatureLength))

    # evaluate 
    liResult = keModel.evaluate(
        oFeatures.arFeatures,
        oFeatures.arLabelsOneHot,
        batch_size = None,
        verbose = 1)
    
    print(keModel.metrics_names)
    print(liResult)

    return


def predict(sFeatureDir, sModelSaved, sClassFile, nFramesNorm, nFeatureLength):    
    """ predict class for all features in given directory on saved model """
    
    print("\nPredict features on LSTM ...")
    # load classes description: nIndex,sClass,sLong,sCat,sDetail
    dfClass = pd.read_csv(sClassFile, header = 0, index_col = 0, dtype = {"sClass":str})
    
    # Load features
    oFeatures = Features(sFeatureDir, nFramesNorm, nFeatureLength)

    # load model from file
    print("Loading saved LSTM neural network %s ..." % sModelSaved)
    keModel = load_model(sModelSaved)
    assert(keModel.input_shape == (None, nFramesNorm, nFeatureLength))

    # predict
    arProba = keModel.predict(
        oFeatures.arFeatures, 
        batch_size = None, 
        verbose = 1
    )
    arPred = arProba.argmax(axis=1)
    
    # compare
    #print("Groundtruth:", oFeatures.liLabels)
    #print("Predicted:  ", arPred)
    #print("Predicted:  ", dfClass.sClass[arPred])
    print("Accuracy: {:.3f}".format(np.mean(oFeatures.liLabels == dfClass.sClass[arPred])))

    return


def main():
   
    # directories
    sClassFile = "datasets/04-chalearn/class.csv"
    sVideoDir = "datasets/04-chalearn"
    sFrameDir = "03a-chalearn/data/frame-20"
    sFeatureDir = "03a-chalearn/data/feature-mobilenet"
    sModelDir = "03a-chalearn/model"
    #sModelSaved = sModelDir + "/20180525-1033-lstm-35878in249-best.h5"
    sLogDir = "03a-chalearn/log"

    nClasses = 3
    nFramesNorm = 20 # number of frames per video for feature calculation
    sConvNet = "mobilenet"

    print("\nStarting ChaLearn extraction & train in directory:", os.getcwd())

    # extract frames from videos
    #video2frames(sVideoDir, sFrameDir, nFramesNorm, nClasses)
    
    # calculate features from frames
    #frames2features(sConvNet, sFrameDir, sFeatureDir, nFramesNorm, nClasses)

    # train the LSTM network
    sClassFile, sModelSaved = train(sConvNet, sFeatureDir, None, sModelDir, sLogDir, 
        nFramesNorm, nBatchSize=256, nEpoch=3, fLearn=1e-3)

    # evaluate features on LSTM
    #evaluate(sFeatureDir + "/val", sModelSaved, nFramesNorm, nFeatureLength)

    # predict labels from features
    #predict(sFeatureDir + "/val", sModelSaved, sClassFile, nFramesNorm, nFeatureLength)

if __name__ == '__main__':
    main()
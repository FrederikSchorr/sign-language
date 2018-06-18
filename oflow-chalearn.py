import os
import time

from s3chalearn.preprocess import video2frames
from s6opticalflow.preprocess import framesDir2flowsDir
from s3chalearn.deeplearning import ConvNet, RecurrentNet, VideoClasses 
from s3chalearn.feature_train_test import frames2features, train, evaluate, predict



def main():
   
    nClasses = 2 # number of classes
    nFramesNorm = 20 # number of frames per video for feature calculation

    # directories
    sClassFile   = "data-set/04-chalearn/class.csv"
    sVideoDir    = "data-set/04-chalearn"
    sFrameDir    = "data-temp/04-chalearn/%03d-frame-20"%(nClasses)
    sFlowDir     = "data-temp/04-chalearn/%03d-oflow-20"%(nClasses)
    sFeatureDir  = "data-temp/04-chalearn/%03d-oflow-20-mobilenet"%(nClasses)
    sModelDir    = "model"
    sLogPath     = "log/" + time.strftime("%Y%m%d-%H%M") + \
        "chalearn%03d-oflow-lstm.csv"%(nClasses)

    print("\nStarting ChaLearn with optical flow and mobilenet+lstm in directory:", os.getcwd())

    # unzip ChaLearn videos and sort them in folders=label
    #unzip_sort_videos(sVideoDir, sVideoDir + "/_zip/train.zip", sVideoDir + "/_zip/train.txt")
    #unzip_sort_videos(sVideoDir, sVideoDir + "/_zip/val.zip", sVideoDir + "/_zip/val.txt")

    # extract frames from videos
    video2frames(sVideoDir, sFrameDir, nFramesNorm, nClasses)

    # calculate optical flow
    framesDir2flowsDir(sFrameDir, sFlowDir)

    # calculate features from frames
    oCNN = ConvNet("mobilenet")
    oCNN.load_model()
    frames2features(sFrameDir, sFeatureDir, oCNN, nFramesNorm, nClasses)

    # train the LSTM network
    oClasses = VideoClasses(sClassFile)
    oRNN = RecurrentNet("lstm", nFramesNorm, oCNN.nOutputFeatures, oClasses)
    oRNN.build_compile(fLearn=1e-3)
    #oRNN.load_model(sModelSaved)
    oRNN = train(sFeatureDir, sModelDir, sLogPath, oRNN, 
        nBatchSize=256, nEpoch=100)

    # predict labels from features
    predict(sFeatureDir + "/val", oRNN)

if __name__ == '__main__':
    main()
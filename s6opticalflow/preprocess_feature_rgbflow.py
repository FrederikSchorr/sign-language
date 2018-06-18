import os
import time
import sys

sys.path.append(os.path.abspath("."))
from s3chalearn.preprocess import video2frames
from s6opticalflow.preprocess import framesDir2flowsDir
from s3chalearn.deeplearning import ConvNet, RecurrentNet, VideoClasses 
from s3chalearn.feature_train_predict import frames2features, train, evaluate, predict


def main():
   
    nClasses = 3 # number of classes
    nFramesNorm = 20 # number of frames per video for feature calculation

    # directories
    sClassFile       = "data-set/04-chalearn/class.csv"
    sVideoDir        = "data-set/04-chalearn"
    sFrameDir        = "data-temp/04-chalearn/%03d-frame-20"%(nClasses)
    sFlowDir         = "data-temp/04-chalearn/%03d-oflow-20"%(nClasses)
    sFrameFeatureDir = "data-temp/04-chalearn/%03d-frame-20-mobilenet"%(nClasses)
    sFlowFeatureDir  = "data-temp/04-chalearn/%03d-oflow-20-mobilenet"%(nClasses)

    sLogPath         = "log/" + time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-chalearn%03d-lstm.csv"%(nClasses)

    print("\nPreprocess ChaLearn frames + optical flow and calculate mobilenet features. \nDirectory:", os.getcwd())

    # extract frames from videos
    video2frames(sVideoDir, sFrameDir, nFramesNorm, nClasses)

    # calculate features from frames
    oCNN = ConvNet("mobilenet")
    oCNN.load_model()
    frames2features(sFrameDir, sFrameFeatureDir, oCNN, nFramesNorm, nClasses)

    # calculate optical flow
    framesDir2flowsDir(sFrameDir, sFlowDir)

    # calculate features from optical flow
    frames2features(sFlowDir, sFlowFeatureDir, oCNN, nFramesNorm, nClasses)

    return


if __name__ == '__main__':
    main()
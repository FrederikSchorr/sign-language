import os
import time
import sys

sys.path.append(os.path.abspath("."))
from s7i3d.preprocess import videosDir2framesDir
from s6opticalflow.preprocess import framesDir2flowsDir


def main():
   
    nClasses = 249 # number of classes

    # directories
    sClassFile       = "data-set/04-chalearn/class.csv"
    sVideoDir        = "data-set/04-chalearn"
    sFrameDir        = "data-temp/04-chalearn/%03d-frame"%(nClasses)
    sFlowDir         = "data-temp/04-chalearn/%03d-oflow"%(nClasses)

    sLogPath         = "log/" + time.strftime("%Y%m%d-%H%M", time.gmtime()) + \
        "-chalearn%03d-lstm.csv"%(nClasses)

    print("\nPreprocess ChaLearn frames + optical flow. \nDirectory:", os.getcwd())

    # extract frames from videos
    videosDir2framesDir(sVideoDir, sFrameDir, nClasses)

    # calculate optical flow
    framesDir2flowsDir(sFrameDir, sFlowDir)

    return


if __name__ == '__main__':
    main()
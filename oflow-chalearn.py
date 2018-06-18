from a_chalearn import videounzip, preprocess


def main():
   
    nClasses = 3 # number of classes
    nFramesNorm = 20 # number of frames per video for feature calculation

    # directories
    sClassFile = "data-set/04-chalearn/class.csv"
    sVideoDir = "data-set/04-chalearn"
    sFrameDir = "data-temp/04-chalearn/%03d-frame-20"%(nClasses)

    print("\nStarting ChaLearn with optical flow and mobilenet+lstm in directory:", os.getcwd())

    # unzip ChaLearn videos and sort them in folders=label
    #unzip_sort_videos(sVideoDir, sVideoDir + "/_zip/train.zip", sVideoDir + "/_zip/train.txt")
    #unzip_sort_videos(sVideoDir, sVideoDir + "/_zip/val.zip", sVideoDir + "/_zip/val.txt")

    # extract frames from videos
    video2frames(sVideoDir, sFrameDir, nFramesNorm, nClasses)

if __name__ == '__main__':
    main()
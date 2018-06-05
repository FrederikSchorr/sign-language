"""
Classify 1 video
a. Extract image frames
b. Extract features for each fram using Inception V3
c. Classify with trained LSTM
"""
import numpy as np

import glob
import os
import time
from subprocess import call
from sys import getsizeof

from extractor import Extractor
from models import ResearchModels

# number of frames per video
frames_nb = 40

# features per frame
features_length = 2048

g_timer_total = 0.0
def timer_start():
    global g_timer_start
    g_timer_start = time.time()
    return g_timer_start

def timer_stop():
    delta = time.time()-g_timer_start
    global g_timer_total
    g_timer_total += delta
    print("Timer: %3.2f sec" % delta)
    return delta

def timer_total():
    global g_timer_total
    print("Total prediction time: %3.2f sec" % g_timer_total)
    g_timer_total = 0.0


def video_to_frames(path, videofile):
    """ Extract frames from video
    
    return frame filenames as a list
    """

    src = os.path.join(path, videofile)
    print("Video to classify:", src)
    filename_no_ext = videofile.split('.')[0]

    # create new directory
    dest_path = os.path.join(path, filename_no_ext)
    if not os.path.exists(dest_path):
        os.mkdir(dest_path)
    print("Created directory", dest_path)

    # invoke `ffmpeg -i video.mpg image-%04d.jpg` to exctract images
    dest = os.path.join(dest_path, filename_no_ext + '-%04d.jpg')
    timer_start()
    call(["ffmpeg", "-i", src, dest])
    timer_stop()

    # return list with all image files
    frames = glob.glob(os.path.join(dest_path, "*.jpg"))
    print("Extracted %d frames from video" % len(frames))

    return frames
    

def frames_to_features(frames):
    """ Extract InceptionV3 features from all images in path

    return list with extracted features
    """
    # ensure max number of frames
    if (len(frames) > frames_nb):
        print("Only using first %d of %d frames" % (frames_nb, len(frames)))
        frames = frames[:frames_nb]
    
    # get the model
    print("Load Inception v3 network ...")
    cnn = Extractor()

    # loop thru    
    sequence = []
    timer_start()
    for image in frames:
        print("Extracting features from", image)
        features = cnn.extract(image)
        sequence.append(features)
    timer_stop()
    
    print("Extracted features from %d frames" % len(frames))
    return sequence


def classify_video(lstm_weights, frames_features):
    """ Run the features (of the frames) through LSTM to classify video

    Returns most probable video-category
    """
    print("Load LSTM network ...")
    lstm = ResearchModels(model="lstm", saved_model=lstm_weights, nb_classes=101, seq_length=frames_nb)
    
    # Resize features for input into lstm model
    X = np.array(frames_features)
    X.resize(1, frames_nb, features_length)

    print("Predict video category ...")
    timer_start()
    category = lstm.model.predict(X)
    timer_stop()
    print("Most probable 3 categories:", category.argsort(axis=1)[:,-3:][:,::-1])
    
    return category


def main():
    path = "data/demo"
    videofile = "v_BlowDryHair_g01_c02.avi"
    lstm_weights = 'data/checkpoints/lstm-features.015-1.221.hdf5'

    frames = video_to_frames(path, videofile)
    frames_features = frames_to_features(frames)
    category = classify_video(lstm_weights, frames_features)

    timer_total()

if __name__ == '__main__':
    main()
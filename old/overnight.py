from prepare_mobile import prepare_mobile
from train_mobile_lstm import train_mobile_lstm

"""diVideoSet = {"sName" : "ledasila",
    "nClasses" : 21,   # number of classes
    "nFramesNorm" : 64,    # number of frames per video
    "nMinDim" : 240,   # smaller dimension of saved video-frames
    "tuShape" : (288, 352), # height, width
    "nFpsAvg" : 25,
    "nFramesAvg" : 75,
    "fDurationAvg" : 3.0} # seconds

"""
diVideoSet = {"sName" : "chalearn",
    "nClasses" : 20,   # number of classes
    "nFramesNorm" : 40,    # number of frames per video
    "nMinDim" : 240,   # smaller dimension of saved video-frames
    "tuShape" : (240, 320), # height, width
    "nFpsAvg" : 10,
    "nFramesAvg" : 50, 
    "fDurationAvg" : 5.0} # seconds 


prepare_mobile(diVideoSet, bImage = True, bOflow = False)
train_mobile_lstm(diVideoSet, bImage = True, bOflow = False)
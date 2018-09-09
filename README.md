# Sign Language Recognition

* This prototype "understands" sign language for deaf people
* Includes all code to prepare data (eg from ChaLearn dataset), extract features, train neural network, and predict signs during live demo
* Based on deep learning techniques, in particular convolutional neural networks (including state-of-the-art 3D model) and recurrent neural networks (LSTM)
* Built with Python, Keras+Tensorflow and OpenCV (for video capturing and manipulation) 

For 10-slide presentation + 1-min demo video see [here](https://docs.google.com/presentation/d/1KSgJM4jUusDoBsyTuJzTsLIoxWyv6fbBzojI38xYXsc/edit?usp=sharing).

## Requirements

This code requires at least
* python 3.6.5
* tensorflow 1.8.0
* keras 2.2.0
* opencv-python 3.4.1.15

For the training of the neural networks a GPU is necessary (eg aws p2.xlarge). The live demo works on an ordinary laptop (without GPU), eg MacBook Pro, i5, 8GB.
  
## Get the video data

See here for overview of suitable data-sets for sign-language for deaf people: https://docs.google.com/presentation/d/1KSgJM4jUusDoBsyTuJzTsLIoxWyv6fbBzojI38xYXsc/edit#slide=id.g3d447e7409_0_0

Download the ChaLearn Isolated Gesture Recognition dataset here: http://chalearnlap.cvc.uab.es/dataset/21/description/ (you need to register first)

The ChaLearn video descriptions and labels (for train, validation and test data) can be found here: [data_set/chalearn](https://github.com/FrederikSchorr/sign-language/tree/master/data-set/chalearn/_download)

[prepare_chalearn.py](prepare_chalearn.py) is used to unzip the videos and sort them by labels (using Keras best-practise 1 folder = 1 label): ![folderstructure](https://github.com/FrederikSchorr/sign-language/blob/master/image/readme_folderstructure.jpg)


## Prepare the video data 

### Extract image frames from videos
[frame.py](frame.py) extracts image frames from each video (using OpenCV) and stores them on disc.

See [pipeline_i3d.py](pipeline_i3d.py) for the parameters used for the ChaLearn dataset:
* 40 frames per training/test videos (on average 5 seconds duration = approx 8 frames per second)
* Frames are resized/cropped to 240x320 pixels

### Calculate optical flow
[opticalflow.py](opticalflow.py) calculates optical flow from the image frames of a video (and stores them on disc). See [pipeline_i3d.py](pipeline_i3d.py) for usage.

Optical flow is very effective for this type of video classification, but also very calculation intensive, see [here](https://docs.google.com/presentation/d/1KSgJM4jUusDoBsyTuJzTsLIoxWyv6fbBzojI38xYXsc/edit#slide=id.g3d3364860a_0_122).


## Train the neural network
[train_i3d.py](train_i3d.py) trains the neural network. First only the (randomized) top layers are trained, then the entire (pre-trained) network is fine-tuned.

A pre-trained 3D convolutional neural network, I3D, developed in 2017 by Deepmind is used, see [here](https://docs.google.com/presentation/d/1KSgJM4jUusDoBsyTuJzTsLIoxWyv6fbBzojI38xYXsc/edit#slide=id.g3d3364860a_0_169) and [model_i3d.py](model_i3d.py). 

Training requires a GPU and is performed through a generator which is provided in [datagenerator.py](datagenerator.py).

*Note: the code files containing "_mobile_lstm" are used for an alternative NN architecture, see [here](https://docs.google.com/presentation/d/1KSgJM4jUusDoBsyTuJzTsLIoxWyv6fbBzojI38xYXsc/edit#slide=id.g3d3364860a_0_27).*


## Predict during live demo

[livedemo.py](livedemo.py) launches the webcam, 
* waits for the start signal from user,
* captures 5 seconds of video (using [videocapture.py](videocapture.py)),
* extracts frames from the video
* calculates and displays the optical flow,
* and uses the neural network to predict the sign language gesture.

The neural network model is not included in this GitHub repo (too large) but can be downloaded [here](https://drive.google.com/open?id=165fKeQY1AhbMUVnV8MyQrMnNWbO7d3fg) (150 MB).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


## Acknowledgments

* Inspired by Matt Harveys blog post + repository: *Five video classification methods implemented in Keras and TensorFlow* (Mar 2017)
    * https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5
    * https://github.com/harvitronix/five-video-classification-methods
* [Oscar Koller](https://www-i6.informatik.rwth-aachen.de/~koller/) from RWTH Aachen provided an excellent overview of state-of-the-art research on sign language recognition, including his paper on *Sign Language Translation* (2018) https://www-i6.informatik.rwth-aachen.de/publications/download/1064/Camgoz-CVPR-2018.pdf
* The I3D video classification model was introduced in: *Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset* (2017), Joao Carreira, Andrew Zisserman (both Google Deepmind), https://arxiv.org/abs/1705.07750v1
* Keras implementation of above I3D model (Jan 2018): https://github.com/dlpbc/keras-kinetics-i3d
* ChaLearn dataset by Barcelona University, 249 (isolated) human gestures, 50.000 videos:
http://chalearnlap.cvc.uab.es/dataset/21/description/
* This project was developed during the spring 2018 [Data Science Retreat](https://www.datascienceretreat.com/), [Tristan Behrens](http://ai-guru.de/) and [Belal Chaudhary](https://github.com/BelalC) provided valuable coaching.

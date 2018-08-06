# Sign Language Recognition

* This prototype "understands" sign language for deaf people
* Includes all code to prepare data (from ChaLearn dataset), extract features, train neural network, and predict signs during live demo
* Based on deep learning techniques, in particular convolutional neural networks (including state-of-the-art 3D model) and recurrent neural networks (LSTM)
* Built with Python, Keras+Tensorflow and OpenCV (for video capturing and manipulation) 

## Requirements

This code requires at least
* python 3.6.5
* tensorflow 1.8.0
* keras 2.2.0
* opencv-python 3.
## Getting the data

See here for overview of suitable data-sets for sign-language for deaf people: https://docs.google.com/presentation/d/1KSgJM4jUusDoBsyTuJzTsLIoxWyv6fbBzojI38xYXsc/edit#slide=id.g3d447e7409_0_0

Download the ChaLearn Isolated Gesture Recognition dataset here: http://chalearnlap.cvc.uab.es/dataset/21/description/ (you need to register first)

The ChaLearn video descriptions and labels (for train, validation and test data) can be found here: [data_set/chalearn](https://github.com/FrederikSchorr/sign-language/tree/master/data-set/chalearn/_download)

You can use [prepare_chalearn.py](prepare_chalearn.py) to unzip the videos and sort them by labels (using Keras best-practise 1 folder = 1 label).

The resulting folder structure may then look similar to ![this](https://drive.google.com/open?id=1yvXylPyLJAPaxwE1RIFh3QpYDw2m1pL7).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Inspired by Matt Harveys blog post + repository: *Five video classification methods implemented in Keras and TensorFlow* (Mar 2017)
    * https://blog.coast.ai/five-video-classification-methods-implemented-in-keras-and-tensorflow-99cad29cc0b5
    * https://github.com/harvitronix/five-video-classification-methods
* The video classification I3D model is introduced in: *Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset* (2017), Joao Carreira, Andrew Zisserman (both Google Deepmind), https://arxiv.org/abs/1705.07750v1
* Keras implementation of above I3D model (Jan 2018): https://github.com/dlpbc/keras-kinetics-i3d
* ChaLearn dataset by Barcelona University, 249 (isolated) human gestures, 50.000 videos:
http://chalearnlap.cvc.uab.es/dataset/21/description/

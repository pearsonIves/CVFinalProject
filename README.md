# CVFinalProject

# Abstract
As humans, we take for granted our ability to recognize one another. While very simple to us, it is in reality a quite complex task for a machine. We sought to use modern methods of facial detection and image recognition to create a model that could learn to recognize the faces of people and use their faces to determine their identity. While not a perfect result, after many training sessions we were able to train a model with 76.19% accuracy. This is above our goal accuracy of 50%, and we believe is a strong result for a first time project in facial recognition and identification.

# Introduction
We hoped to use facial detection in combination with modern methods of machine learning to create a face recognition system. This functionality could be useful for a wide variety of reasons, including face identification, face authentication, image indexing, and more. We hoped that by combining facial detection and feature extraction functionality into one program, that we could create a more accurate program than one that did not detect faces before extracting a person's features. We feel that the value in our project is in the combination of existing ideas into one system, such as Haar Cascades for face detection, Convolutional Neural Networks for feature extraction, and a Siamese implementation of said CNNs for more accurate results in a limited dataset. 

## Approach
Our approach can be separated into three main parts, a general approach to detecting and identifying faces , and our specific approaches to training our CNN and Siamese CNN models.

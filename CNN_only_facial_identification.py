"""
# Filename: CNN_only_facial_identification.py
# Authors: Pearson Ives and Saksham Raina
# Date: 11/21/2020
# Description: This file contains a face identification system. It uses google's tensorflow and Keras
#               as well as openCV's haar cascades to extract faces and train a convolutional neural network
#               for facial identification. Our implementation trained on the five faces data set from Kaggle.
"""

import cv2
import os
from google.colab.patches import cv2_imshow
import tensorflow as tf
import numpy as np

from tensorflow import expand_dims
from keras import datasets
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense, SpatialDropout2D, Input, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

import matplotlib.pyplot as plt

people = ["ben_afflek", "elton_john", "jerry_seinfeld", "madonna", "mindy_kaling"]
inputShape = (128, 128, 3)
path = "/content/drive/My Drive/Kaggle/CVFinalData/"

def initFaceCascades():
  faceCascades = []

  # Set paths to all haar and lbp cascades
  haarFacesPath = "/content/drive/My Drive/Kaggle/CVFinalData/haarcascades/haarcascade_frontalface_alt2.xml"
  haarFaces2Path = "/content/drive/My Drive/Kaggle/CVFinalData/haarcascades/haarcascade_frontalface_alt.xml"
  haarFaces3Path = "/content/drive/My Drive/Kaggle/CVFinalData/haarcascades/haarcascade_profileface.xml"
  haarFaces4Path = "/content/drive/My Drive/Kaggle/CVFinalData/haarcascades/haarcascade_frontalface_default.xml"
  haarFaces5Path = "/content/drive/My Drive/Kaggle/CVFinalData/haarcascades/haarcascade_frontalface_alt_tree.xml"

  lbpcascadesPath = "/content/drive/My Drive/Kaggle/CVFinalData/lbpcascades/lbpcascade_frontalface.xml"
  lbpcascades2Path = "/content/drive/My Drive/Kaggle/CVFinalData/lbpcascades/lbpcascade_frontalface_improved.xml"
  lbpcascades3Path = "/content/drive/My Drive/Kaggle/CVFinalData/lbpcascades/lbpcascade_profileface.xml"

  # create cascade classifiers
  faceCascade = cv2.CascadeClassifier(haarFacesPath)
  faceCascades.append(faceCascade)

  faceCascade2 = cv2.CascadeClassifier(haarFaces2Path)
  faceCascades.append(faceCascade2)

  faceCascade3 = cv2.CascadeClassifier(haarFaces3Path)
  faceCascades.append(faceCascade3)

  faceCascade4 = cv2.CascadeClassifier(haarFaces4Path)
  faceCascades.append(faceCascade4)

  faceCascade5 = cv2.CascadeClassifier(haarFaces5Path)
  faceCascades.append(faceCascade5)

  faceCascade6 = cv2.CascadeClassifier(lbpcascadesPath)
  faceCascades.append(faceCascade6)

  faceCascade7 = cv2.CascadeClassifier(lbpcascades2Path)
  faceCascades.append(faceCascade7)

  faceCascade8 = cv2.CascadeClassifier(lbpcascades3Path)
  faceCascades.append(faceCascade8)

  return faceCascades

def detectFacesAndGenerateTrainSet(path, faceCascades):
   #For each celebrity
  data = {}
   #Compile list of regions of interest to insert into 
  roiList = []
  #keep track of labels
  # "ben_afflek", "elton_john", "jerry_seinfeld", "madonna", "mindy_kaling"
  # 1 = ben afflek, 2 = elton john, 3 = jerry seinfeld, 4 = madonna, 5 = mindy kaling
  roiLabels = []
  for person in people:
    data[person] = []
    numDetected = 0
    newPath = path + str(person)
    #path = "/content/drive/My Drive/CVFinalData/train/" + str(person)
    print(path)
    
    for f in os.listdir(newPath):

      imagePath = newPath + '/' + str(f)
      #print(imagePath)
      input = cv2.imread(imagePath)
      faces = []
      i = 0
      
      
      # Find their face. If no face has been detected, attempt to find a face
      # With the next face cascade
      faces = faceCascades[i].detectMultiScale(input,
                                          scaleFactor=1.3,
                                          minNeighbors=5,
                                          minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
      if len(faces) == 0:
        i += 1
        faces = faceCascades[i].detectMultiScale(input,
                                          scaleFactor=1.3,
                                          minNeighbors=5,
                                          minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
      if len(faces) == 0:
        i += 1
        faces = faceCascades[i].detectMultiScale(input,
                                          scaleFactor=1.4,
                                          minNeighbors=5,
                                          minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
      if len(faces) == 0:
        i += 1
        faces = faceCascades[i].detectMultiScale(input,
                                          scaleFactor=1.4,
                                          minNeighbors=5,
                                          minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
      if len(faces) == 0:
        i += 1
        faces = faceCascades[i].detectMultiScale(input,
                                          scaleFactor=1.3,
                                          minNeighbors=5,
                                          minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
      if len(faces) == 0:
        i += 1
        faces = faceCascades[i].detectMultiScale(input,
                                          scaleFactor=1.4,
                                          minNeighbors=5,
                                          minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
      if len(faces) == 0:
        i += 1
        faces = faceCascades[i].detectMultiScale(input,
                                          scaleFactor=1.4,
                                          minNeighbors=5,
                                          minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
      if len(faces) == 0:
        i += 1
        faces = faceCascades[i].detectMultiScale(input,
                                          scaleFactor=1.4,
                                          minNeighbors=5,
                                          minSize=(60, 60),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

      #print(faces)
      
     
      for (x, y, w, h) in faces:
          #isolate results
          regionOfInterest = input[y:y + h, x:x + w]
          

          numDetected += 1
          
          #Add labels
          label = 0
          if person == "ben_afflek": label = 0;
          elif person == "elton_john": label = 1;
          elif person == "jerry_seinfeld": label = 2;
          elif person == "madonna": label = 3;
          elif person == "mindy_kaling": label = 4;
          roiLabels.append(label)
          #print("labeled as " + str(label) + ":")


          #show results
          resizedImage = cv2.resize(regionOfInterest, (128,128))
          roiList.append(resizedImage)
          #cv2_imshow(resizedImage)
          data[person].append(resizedImage)
          cv2.rectangle(input, (x, y), (x + w, y + h), (0, 255, 0), 2)
          #cv2_imshow(input)
    

    print("Number of faces detected for ", person, ": ", numDetected)
  return data, roiList, roiLabels

def generateBatch(data, batchSize = 20):
  pairs=[np.zeros((batchSize, 128, 128, 3)) for i in range(2)]

  targets=np.zeros((batchSize,))

  # make half the batch different class and other half the same
  targets[batchSize//2:] = 1
  for i in range(batchSize):
      # for each of 20 pictures, we are picking a random person to include in batch
      person = people[np.random.randint(0,5)]

      # if we have 5 images of ben
      # index is rand(0,5)
      idx1 = np.random.randint(0, len(data[person]))
      pairs[0][i,:,:,:] = data[person][idx1].reshape(128, 128, 3)
      
      # pick images of same class for 1st half, different for 2nd
      # batchSize //2 = 10
      if i <= batchSize // 2:
          person2 = person
      else: 
          # add a random number to the category modulo n classes to ensure 2nd image has a different category
          person2 = people[np.random.randint(0,5)]
          while (person2 == person):
            person2 = people[np.random.randint(0,5)]

      idx2 = np.random.randint(0, len(data[person2])) 
      pairs[1][i,:,:,:] = data[person2][idx2].reshape(128, 128, 3)

      #making a batch. first ten people in each half of the pair are the same (as the one in the other half),
      # second ten people in each half are different people than the corresponding person in the other half at the same index
      # at index 0, pair[0][0] is different than pair[1][0]
      # [     [image , image, image.....], [image, image, image....]    ]
  
  return pairs, targets

def initializeWeights(shape, dtype=None):
  return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initializeBias(shape, dtype=None):
  return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def getCNNModelEncoding():
    
    inputShape = (128, 128, 3)
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(32, (7,7), input_shape=inputShape,
                     kernel_initializer=initializeWeights,
                     bias_initializer=initializeBias, kernel_regularizer=l2(2e-4)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2,2)))
    #32 kernels of 5x5
    model.add(Conv2D(32, (5,5),
                     kernel_initializer=initializeWeights,
                     bias_initializer=initializeBias, kernel_regularizer=l2(2e-4)))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2,2)))
    #64 kernels of 3x3
    model.add(Conv2D(64, (3,3), kernel_initializer=initializeWeights,
                     bias_initializer=initializeBias, kernel_regularizer=l2(2e-4)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Conv2D(32, (4,4), kernel_initializer=initializeWeights,
                    bias_initializer=initializeBias, kernel_regularizer=l2(2e-4)))
    model.add(Activation('relu'))
  
    #fully connected layer with 32 neurons 
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    
    #Add last 
    model.add(Dense(5))
    
    # return the model
    return model



def predictTestData(testData, cnn, featureVectors):
  correct = 0
  false = 0
  for person in people:
    for face in testData[person]:
      newImage = expand_dims(face,0)
      newFeatVector = cnn.predict(newImage)
      minDistance = float("inf")
      bestMatch = ""
      for key, value in featureVectors.items():
        #print(newFeatVector)
        distance = np.linalg.norm(newFeatVector[0] - value[0])
        #print(distance)
        if distance < minDistance:
          minDistance = distance
          bestMatch = key

      if bestMatch == person:
        correct+=1
      else:
        false+=1
  print("Accuracy: ")
  print(correct/(correct+false))
  return correct, false

if __name__ == "__main__":
  
  class_names = ['ben_afflek', 'elton_john',"jerry_seinfeld", "madonna",  "mindy_kaling"]
 
  
  
  faceCascades = initFaceCascades()
  trainPath = path + "train/"
  unused, train_images, train_labels = detectFacesAndGenerateTrainSet(trainPath, faceCascades)
  train_images = [x  / 255.0 for x in train_images]
  
  
  train_images = np.array(train_images)
  train_labels = np.array(train_labels)
  
  testPath = path + "val/"
  unused, test_images, test_labels = detectFacesAndGenerateTrainSet(testPath, faceCascades)
  test_images = [x  / 255.0 for x in test_images]
  
  test_labels = np.array(test_labels)
  test_images = np.array(test_images)


  model = getCNNModelEncoding()
  optimizer = Adam(lr = 0.0007)
  model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  history = model.fit(train_images, train_labels, epochs=100, validation_data=(test_images, test_labels) )

  plt.plot(history.history['accuracy'], label='accuracy')
  plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.ylim([0.0, 1])
  plt.legend(loc='lower right')
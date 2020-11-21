"""
# Filename: Siamese_CNN_facial_identification.py
# Authors: Pearson Ives and Saksham Raina
# Date: 11/21/2020
# Description: This file contains a face identification system. It uses google's tensorflow and Keras
#               as well as openCV's haar cascades to extract faces and train a convolutional neural network
#               for facial identification. Our implementation trained on the five faces data set from Kaggle.
#               It uses a Siamese CNN implementation to identify faces.
"""
import cv2
import os
from google.colab.patches import cv2_imshow

import numpy as np

from tensorflow import expand_dims

from keras.datasets import mnist
from keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense, SpatialDropout2D, Input, Lambda
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

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
      for faceCascade in faceCascades:
        for scale in np.linspace(1.1, 1.4, 4):
          faces = faceCascade.detectMultiScale(input,
                                          scaleFactor=scale,
                                          minNeighbors=10,
                                          flags=cv2.CASCADE_SCALE_IMAGE)
          if len(faces) > 0: break

        if len(faces) > 0: break
      
      
      
      
      #Compile list of regions of interest to insert into 
      roiList = []

      #keep track of labels
      # "ben_afflek", "elton_john", "jerry_seinfeld", "madonna", "mindy_kaling"
      # 1 = ben afflek, 2 = elton john, 3 = jerry seinfeld, 4 = madonna, 5 = mindy kaling
      roiLabels = []
      for (x, y, w, h) in faces:
          #cv2_imshow(input)
          #isolate results
          regionOfInterest = input[y:y + h, x:x + w]
          roiList.append(regionOfInterest)

          numDetected += 1
          
          #Add labels
          label = 0
          if person == "ben_afflek": label = 1;
          elif person == "elton_john": label = 2;
          elif person == "jerry_seinfeld": label = 3;
          elif person == "madonna": label = 4;
          elif person == "mindy_kaling": label = 5;
          roiLabels.append(label)
          #print("labeled as " + str(label) + ":")


          #show results
          resizedImage = cv2.resize(regionOfInterest, (128,128))
          data[person].append(resizedImage)
          cv2.rectangle(input, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print("Number of faces detected for ", person, ": ", numDetected)
  return data

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

def getCNNModelEncoding(inputShape = inputShape):

  inputShape = (128, 128, 3)
  # Convolutional Neural Network
  model = Sequential()
  model.add(Conv2D(32, 3, input_shape=inputShape,
                    kernel_initializer=initializeWeights,
                    bias_initializer=initializeBias, kernel_regularizer=l2(2e-4)))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(MaxPool2D(pool_size=(2,2)))
  #64 kernels of 5x5
  model.add(Conv2D(32, 3,
                    kernel_initializer=initializeWeights,
                    bias_initializer=initializeBias, kernel_regularizer=l2(2e-4)))
  model.add(Dropout(0.2))
  model.add(MaxPool2D(pool_size=(2,2)))
  #128 kernels of 3x3
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
    
  # return the model
  return model

def siameseCNNMatching(model, inputShape = (128, 128, 3)):
  # Define the tensors for the two input images
    leftInput = Input(inputShape)
    rightInput = Input(inputShape)
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(leftInput)
    encoded_r = model(rightInput)
    
    
    L2Layer = Lambda(lambda tensors:K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=-1, keepdims=True)))
    L2Distance = L2Layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer=initializeBias)(L2Distance)
    
    # Connect the inputs with the outputs
    siameseNet = Model(inputs=[leftInput,rightInput],outputs=prediction)

    return siameseNet

def getAverageCelebFeatureVectors(cnn, data):
  featureVectors = {}
  for person in people:
    featureVector = np.zeros((1,32))
    for face in data[person]:
      newImage = expand_dims(face,0)
      featureVector += cnn.predict(newImage)
    featureVectors[person] = featureVector/len(data[person])

  return featureVectors

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
      print("Predict: ", bestMatch, ", Actual: ", person)
      cv2_imshow(face)

      if bestMatch == person:
        correct+=1
      else:
        false+=1
  print("Accuracy: ")
  print(correct/(correct+false))
  return correct, false

if __name__ == "__main__":
  faceCascades = initFaceCascades()
  trainPath = path + "train/"
  trainData = detectFacesAndGenerateTrainSet(trainPath, faceCascades)
  model = getCNNModelEncoding()
  
  siameseNet = siameseCNNMatching(model)

  optimizer = Adam(lr = 0.0001)
  siameseNet.compile(loss="categorical_hinge", optimizer=optimizer, metrics=["accuracy"])

  batchSize = 100
  nIter = 100 # No. of training iterations / epoch
  model_path = './weights/'


  testPath = path + "val/"
  testData = detectFacesAndGenerateTrainSet(testPath, faceCascades)

  accuracies_l = list()
  correct_l = list()
  total_l = list()
  for i in range(1,nIter+1):
    c = 0
    f = 0
    print("Epoch: {0}/{1}".format(str(i), nIter))
    #if (i % 100) == 0: print(i)
    (inputs,targets) = generateBatch(trainData, batchSize)
    siameseNet.train_on_batch(inputs, targets)
    trainEval = siameseNet.test_on_batch(inputs, targets, return_dict=True )
    print("{0}/{0}".format(batchSize), " [=============] - loss: ", trainEval["loss"], " - accuracy: ", trainEval["accuracy"])
    

  featureVectors = getAverageCelebFeatureVectors(model, trainData)
  predictTestData(testData, model, featureVectors)

  model.save('cnn20000.h5')      
  #siameseNet.save_weights(os.path.join(model_path, 'siameseWeights20000.h5'))


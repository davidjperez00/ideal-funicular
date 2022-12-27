# File Name: multiclass_MLP.py
# Brief: This model trains an MLP neural net on the MNIST dataset
# Date: 10/17/2022
# Author: David Perez
## Gihub: Create python file for visualizing dataset
## LINK HERE

# from tensorflow import keras
import cv2
import os
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from tensorflow import keras
import sys
sys.path.append("/multiclass_MLP_v1_alternate")


# Global varibles
K = keras.backend

# @brief Callback to allow for keras exponential learning rate to increase the 
#        accuracy of the sequential ANN.
class ExponentialLearningRate(keras.callbacks.Callback):    
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.learning_rate))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.learning_rate, self.model.optimizer.learning_rate * self.factor)

def load_add_operator_dataset():

  images_names = os.listdir("./datasets/full_+_preprocessed/")
  add_images = []
  for img_name in images_names:
    image = cv2.imread('./datasets/full_+_preprocessed/'+ img_name, 0)

    add_images.append(image)

  # Convert image to numpy array to match MNIST datatype
  images = np.asarray(add_images)

  # Load labels
  image_labels = create_add_operator_labels()

  # Convert labels to numpy array to match MNIST datatype
  image_labels = np.asanyarray(image_labels)

  # Remove 14% for test
  # X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
  labels_frac = math.ceil(len(image_labels) * (1/7)) # >>> 604
  images_test, images_train = images[:labels_frac] / 255., images[labels_frac:] / 255. 
  labels_test, labels_train = image_labels[:labels_frac], image_labels[labels_frac:]

  # Take another 8% for validation set, remainder stays in train
  labels_frac = math.ceil(len(image_labels) * (.5/6)) # >>> 353
  images_valid, images_train = images[:labels_frac] / 255., images[labels_frac:] / 255. 
  labels_valid, labels_train = image_labels[:labels_frac], image_labels[labels_frac:] 

  return images_train, images_valid, images_test, labels_train, labels_valid, labels_test


# Create float labels for every item in directory
def create_add_operator_labels():
  preprocessed_image_names = os.listdir("./datasets/full_+_preprocessed/")
  add_labels = 10
  y_add_labels = [add_labels for img in range(len(preprocessed_image_names))]

  return np.asarray(y_add_labels)

# @brief Generate data subsets for neural net model.
def create_mnist_train_test():
  # number of images, 28x28 pixels
  # Shape: X:(60000, 28, 28), Y:(10000, 28, 28)
  (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

  # Create validation set and convert to float in 0-1 range
  X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
  y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
  X_test = X_test / 255.

  # Get add operator dataset with labels
  add_train, add_valid, add_test, y_add_train, y_add_valid, y_add_test = load_add_operator_dataset()

  # Add add images to MNIST 
  X_train_new = np.concatenate((X_train, add_train))
  X_test_new = np.concatenate((X_test, add_test))
  X_valid_new = np.concatenate((X_valid, add_valid))
  del add_train, add_valid, add_test

  y_train_new = np.concatenate((y_train, y_add_train))
  y_valid_new = np.concatenate((y_valid, y_add_valid))
  y_test_new = np.concatenate((y_test, y_add_test))
  del y_add_train, y_add_valid, y_add_test

  # Find shuffling algorithm to rearrange the add images and labels
  idx = np.random.permutation(len(y_train_new))
  X_train_new_sh, y_train_new_sh = X_train_new[idx], y_train_new[idx]

  idx = np.random.permutation(len(y_valid_new))
  X_valid_new_sh, y_valid_new_sh = X_valid_new[idx], y_valid_new[idx]

  idx = np.random.permutation(len(y_test_new))
  X_test_new_sh, y_test_new_sh = X_test_new[idx], y_test_new[idx]

  return  X_train_new_sh, X_valid_new_sh, X_test_new_sh, y_train_new_sh, y_valid_new_sh, y_test_new_sh


# @brief Create model object using Keras sequential models.
def create_seq_ANN():
  # Create sequential nueral net model
  model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=[28, 28]),
      keras.layers.Dense(300, activation="relu"),
      keras.layers.Dense(100, activation="relu"),
      keras.layers.Dense(11, activation="softmax")
  ])
  
  return model

# @brief Compiler Keras model using exponential learning rate Keras object
def compile_model(model, lr):
  # Compile the model
  model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=lr),
              metrics=["accuracy"])

  return model

# @brief Provide Keras model with MNIST created datasets
def train_seq_ANN(model, X_train, y_train, X_valid, y_valid, checkpoint_cb, early_stopping_cb):
  # Train model and get history
  history = model.fit(X_train, y_train, epochs=100,
                      validation_data=(X_valid, y_valid),
                      callbacks=[checkpoint_cb, early_stopping_cb])
  return history

# @brief Create an ANN with 98% accuracy on the MNIST datset.
# @TODO change names of models or error will occur
def build_model():
  # Get MIST dataset for ANN multi-classifier
  X_train,  X_valid, X_test, y_train, y_valid, y_test = create_mnist_train_test()

  keras.backend.clear_session()
  np.random.seed(42)
  tf.random.set_seed(42)

  # Create seqential neural net
  model = create_seq_ANN()
  print(model)

  model = compile_model(model, 3e-1)
  print(model)

  early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
  checkpoint_cb = keras.callbacks.ModelCheckpoint("multiclass_MLP_v2_0.h5", save_best_only=True)

  # Train ideal model
  train_seq_ANN(model, X_train, y_train, X_valid, y_valid, checkpoint_cb, early_stopping_cb)

  # saving another version of the model
  model.save('multiclass_MLP_v2')

  # model = keras.models.load_model("my_mnist_model.h5") # rollback to best model
  print(model.evaluate(X_test, y_test))


# @brief Grab the current version model from directory
def get_model():

  X_train,  X_valid, X_test, y_train, y_valid, y_test = create_mnist_train_test()
  model = keras.models.load_model("multiclass_MLP_v1_alternate") # rollback to best model
  
  print(model.evaluate(X_test, y_test))

  return model


build_model()

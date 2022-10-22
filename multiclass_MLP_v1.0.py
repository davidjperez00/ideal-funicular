# File Name: multiclass_MLP.py
# Brief: This model trains an MLP neural net on the MNIST dataset
# Date: 10/17/2022
# Author: David Perez
## Gihub: Create python file for visualizing dataset
## LINK HERE

from tensorflow import keras


def create_mnist_train_test():
  # number of images, 28x28 pixels
  # Shape: X:(60000, 28, 28), Y:(10000, 28, 28)
  (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

  # Create validation set and convert to float in 0-1 range
  X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
  y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
  X_test = X_test / 255.

  return X_train,  X_valid, X_test, y_train, y_valid, y_test

# @brief Create an ANN with 98% accuracy on the MNIST datset.
def build_model():
  # Get MIST dataset for ANN multi-classifier
  X_train,  X_valid, X_test, y_train, y_valid, y_test = create_mnist_train_test()
  print(X_train.shape,  X_valid.shape, X_test.shape, y_train.shape, y_valid.shape, y_test.shape )


build_model()
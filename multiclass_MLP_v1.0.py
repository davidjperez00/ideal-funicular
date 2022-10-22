# File Name: multiclass_MLP.py
# Brief: This model trains an MLP neural net on the MNIST dataset
# Date: 10/17/2022
# Author: David Perez
## Gihub: Create python file for visualizing dataset
## LINK HERE

from tensorflow import keras
import numpy as np
import tensorflow as tf


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

# @brief Generate data subsets for neural net model.
def create_mnist_train_test():
  # number of images, 28x28 pixels
  # Shape: X:(60000, 28, 28), Y:(10000, 28, 28)
  (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()

  # Create validation set and convert to float in 0-1 range
  X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
  y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
  X_test = X_test / 255.

  return X_train,  X_valid, X_test, y_train, y_valid, y_test

# @brief Create model object using Keras sequential models.
def create_seq_ANN():
  # Create sequential nueral net model
  model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=[28, 28]),
      keras.layers.Dense(300, activation="relu"),
      keras.layers.Dense(100, activation="relu"),
      keras.layers.Dense(10, activation="softmax")
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
  checkpoint_cb = keras.callbacks.ModelCheckpoint("multiclass_MLP_v1.0.h5", save_best_only=True)

  # Train ideal model
  train_seq_ANN(model, X_train, y_train, X_valid, y_valid, checkpoint_cb, early_stopping_cb)

  # saving another version of the model
  model.save('multiclass_MLP_v1_alternate')

  # model = keras.models.load_model("my_mnist_model.h5") # rollback to best model
  print(model.evaluate(X_test, y_test))


build_model()

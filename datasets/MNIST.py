# File Name: MNIST.py
# Brief: Main file
# Date: 10/12/2022
# Author: David Perez

from sklearn.datasets import fetch_openml

# Fetching the MNIST dataset
def get_mnist():
  mnist = fetch_openml('mnist_784', version=1, as_frame=False)
  # DESCR key describes the dataset
  # a data key containing an array with on row per instance and one column per feature 
  # a target key containing an array with the labels
  print(mnist.keys())

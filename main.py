# File Name: main.py
# Brief: Main file
# Date: 10/11/2022
# Author: David Perez
# import models/multiclass_MLP_v1_0

# from models import multiclass_MLP_v1_0

from models.multiclass_MLP_v1_0 import get_model
def main():
  model = get_model()

  # @todo handle an inputed image.

  # @todo Convert an image to pdf

  # @todo Identify and number in an image

  # @todo Add data minipulation from user input.
  #     this includes converting an image to pdf.


# WHY DOES THIS RUN TWICE
if __name__ == '__main__':
  main()

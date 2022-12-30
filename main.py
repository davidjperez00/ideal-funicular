# File Name: main.py
# Brief: Main file, creates datatsets and model for entire repo
# Date: 10/11/2022
# Author: David Perez
# import models/multiclass_MLP_v1_0
from datasets import datasets_main
from models import multi_MLP_v2

# from models import multiclass_MLP_v1_0


def main():
  # model = get_model()

  # Create Add operator datasets
  # datasets_main.datasets_init()

  # Create model for repo
  multi_MLP_v2.build_model()



  # @todo handle an inputed image.

  # @todo Convert an image to pdf

  # @todo Identify and number in an image

  # @todo Add data minipulation from user input.
  #     this includes converting an image to pdf.


# WHY DOES THIS RUN TWICE
if __name__ == '__main__':
  main()

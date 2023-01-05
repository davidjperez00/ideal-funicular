# File Name: main.py
# Brief: Main file, creates datatsets and model for entire repo
# Date: 10/11/2022
# Author: David Perez
from datasets import datasets_main
from models import multi_MLP_v2

def main():
  # Create Add operator datasets
  datasets_main.datasets_init()

  # Create model for repo
  multi_MLP_v2.build_model()

if __name__ == '__main__':
  main()

import os

from preprocess_data import *
from NonLinearModel import NonLinearModel
from pathlib import Path

dataPath = os.path.join(os.getcwd(),"Data") + "/"
fileName = "AllFeedData.csv"

if __name__ == "__main__":
  pre_process_data(dataPath, fileName)
  model = NonLinearModel()
  model.fit(dataPath, fileName)

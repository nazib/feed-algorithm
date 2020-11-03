import os
from model.preprocess_data import preprocess_data
from model.NonLinearModel import NonLinearModel

DATA_PATH = os.path.join(os.getcwd(), "Data") + "/"
FILE_NAME = "AllFeedData.csv"

if __name__ == "__main__":
    preprocess_data(DATA_PATH, FILE_NAME)
    model = NonLinearModel()
    model.fit(DATA_PATH, FILE_NAME)

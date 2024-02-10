import os
import urllib
import urllib.request
import cv2
from zipfile import ZipFile

import numpy as np

from nnfs.datasets import sine_data, spiral_data
from scratchnet.models.Model import Model
from scratchnet.layers.layer_dense import Layer_Dense
from scratchnet.layers.layer_dropout import Layer_Dropout
from scratchnet.activation_functions.activation_functions import \
    Activation_ReLU, Activation_Linear, Activation_Sigmoid, Activation_Softmax
from scratchnet.losses.MeanSquaredErrorLoss import Loss_MeanSquaredError
from scratchnet.losses.BinaryCrossEntropyLoss import Loss_BinaryCrossentropy
from scratchnet.optimizers.optimizer_Adam import Optimizer_Adam
from scratchnet.accuracies.RegressionAccuracy import Accuracy_Regression
from scratchnet.accuracies.CategoricalAccuracy import Accuracy_Categorical
from scratchnet.losses.CategoricalCrossEntropyLoss import Loss_CategoricalCrossentropy

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')

# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test

def download_dataset(url : str, file : str, folder : str):

    if not os.path.isfile(file):
        print(f'Downloading {url} and saving as {file}...')
        urllib.request.urlretrieve(url, file)

    print('Unzipping images...')
    with ZipFile(file) as zip_images:
        zip_images.extractall(folder)

if __name__ == "__main__":

    URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
    FILE = 'fashion_mnist_images.zip'
    FOLDER = 'fashion_mnist_images'

    # Download Dataset
    if not os.path.isfile(FOLDER): download_dataset(URL, FILE, FOLDER)

    # Create dataset
    X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

    # Scale features
    X = (X.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5







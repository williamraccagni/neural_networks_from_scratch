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

    fashion_mnist_labels = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'
    FILE = '../download/fashion_mnist_images.zip'
    FOLDER = '../datasets/fashion_mnist_images'

    # Download Dataset
    if not os.path.isdir(FOLDER): download_dataset(URL, FILE, FOLDER)

    # Create dataset
    X, y, X_test, y_test = create_data_mnist('../datasets/fashion_mnist_images')

    # Shuffle the training dataset
    keys = np.array(range(X.shape[0]))
    np.random.shuffle(keys)
    X = X[keys]
    y = y[keys]

    # Scale and reshape samples
    X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    # Instantiate the model
    model = Model()

    # # Add layers
    # model.add(Layer_Dense(X.shape[1], 64))
    # model.add(Activation_ReLU())
    # model.add(Layer_Dense(64, 64))
    # model.add(Activation_ReLU())
    # model.add(Layer_Dense(64, 10))
    # model.add(Activation_Softmax())

    # Add layers
    model.add(Layer_Dense(X.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    # # Set loss, optimizer and accuracy objects
    # model.set(
    #     loss=Loss_CategoricalCrossentropy(),
    #     optimizer=Optimizer_Adam(decay=5e-5),
    #     accuracy=Accuracy_Categorical()
    # )

    # Set loss, optimizer and accuracy objects
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(decay=1e-4),
        accuracy=Accuracy_Categorical()
    )

    # Finalize the model
    model.finalize()

    # Train the model
    model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

    # Retrieve model parameters
    parameters = model.get_parameters()

    # New model

    # Instantiate the model
    model = Model()

    # Add layers
    model.add(Layer_Dense(X.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    # Set loss and accuracy objects
    # We do not set optimizer object this time - there's no need to do it
    # as we won't train the model
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        accuracy=Accuracy_Categorical()
    )

    # Finalize the model
    model.finalize()

    # Set model with parameters instead of training it
    model.set_parameters(parameters)

    # Evaluate the model
    model.evaluate(X_test, y_test)

    confidences = model.predict(X_test[:5])
    print(confidences)
    predictions = model.output_layer_activation.predictions(confidences)
    print(predictions)

    for prediction in predictions:
        print(fashion_mnist_labels[prediction])

    # Test on a draw image


    # Read tshirt image
    image_data = cv2.imread('../datasets/mnist_draw_test_images/tshirt.png', cv2.IMREAD_GRAYSCALE)

    # Resize to the same size as Fashion MNIST images
    image_data = cv2.resize(image_data, (28, 28))

    # Invert image colors
    image_data = 255 - image_data

    # Reshape and scale pixel data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

    # Load the model
    # model = Model.load('fashion_mnist.model')

    # Predict on the image
    predictions = model.predict(image_data)

    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(predictions)

    # Get label name from label index
    prediction = fashion_mnist_labels[predictions[0]]

    print(prediction)


    # Read pants image
    image_data = cv2.imread('../datasets/mnist_draw_test_images/pants.png', cv2.IMREAD_GRAYSCALE)

    # Resize to the same size as Fashion MNIST images
    image_data = cv2.resize(image_data, (28, 28))

    # Invert image colors
    image_data = 255 - image_data

    # Reshape and scale pixel data
    image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

    # Load the model
    # model = Model.load('fashion_mnist.model')

    # Predict on the image
    predictions = model.predict(image_data)

    # Get prediction instead of confidence levels
    predictions = model.output_layer_activation.predictions(predictions)

    # Get label name from label index
    prediction = fashion_mnist_labels[predictions[0]]

    print(prediction)

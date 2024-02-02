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

if __name__ == "__main__":

    # # Create dataset
    # X, y = sine_data()
    #
    # # Instantiate the model
    # model = Model()
    #
    # # Add layers
    # model.add(Layer_Dense(1, 64))
    # model.add(Activation_ReLU())
    # model.add(Layer_Dense(64, 64))
    # model.add(Activation_ReLU())
    # model.add(Layer_Dense(64, 1))
    # model.add(Activation_Linear())
    #
    # # Set loss, optimizer and accuracy objects
    # model.set(
    #     loss=Loss_MeanSquaredError(),
    #     optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
    #     accuracy=Accuracy_Regression()
    # )
    #
    # # Finalize the model
    # model.finalize()
    #
    # # Train the model
    # model.train(X, y, epochs=10000, print_every=100)

    # Create dataset
    X, y = spiral_data(samples=1000, classes=3)
    X_test, y_test = spiral_data(samples=100, classes=3)
    # Instantiate the model
    model = Model()

    # Add layers
    model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4,
                          bias_regularizer_l2=5e-4))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(0.1))
    model.add(Layer_Dense(512, 3))
    model.add(Activation_Softmax())

    # Set loss, optimizer and accuracy objects
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
        accuracy=Accuracy_Categorical()
    )

    # Finalize the model
    model.finalize()

    # Train the model
    model.train(X, y, validation_data=(X_test, y_test),
                epochs=10000, print_every=100)


# IMPORTS
import scratchnet.nn as nn
import numpy as np
import scratchnet.layers.linear as ll
from nnfs.datasets import spiral_data
# visualization
import nnfs
import matplotlib.pyplot as plt
import scratchnet.layers.layer_dense as dense_l
import scratchnet.activation_functions.activation_functions as af
import scratchnet.losses.CategoricalCrossEntropyLoss as ccel
import scratchnet.optimizers.optimizer_SGD as oSGD


if __name__ == '__main__':

    # Training Data with nnfss
    nnfs.init()

    # Create dataset
    X, y = spiral_data(samples=100, classes=3)

    # Create Dense layer with 2 input features and 3 output values
    dense1 = dense_l.Layer_Dense(2, 3)

    # Create ReLU activation (to be used with Dense layer):
    activation1 = af.Activation_ReLU()

    # Create second Dense layer with 3 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = dense_l.Layer_Dense(3, 3)

    # # Create Softmax activation (to be used with Dense layer):
    # activation2 = af.Activation_Softmax()
    #
    # # Create loss function
    # loss_function = ccel.Loss_CategoricalCrossentropy()

    # Create Softmax classifier's combined loss and activation - New part
    loss_activation = af.Activation_Softmax_Loss_CategoricalCrossentropy()

    # Make a forward pass of our training data through this layer
    dense1.forward(X)

    # Make a forward pass through activation function
    # it takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Make a forward pass through second Dense layer
    # it takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)



    # # Make a forward pass through activation function
    # # it takes the output of second dense layer here
    # activation2.forward(dense2.output)
    #
    # # Let's see output of the first few samples:
    # print(activation2.output[:5])
    #
    # # Perform a forward pass through loss function
    # # it takes the output of second dense layer here and returns loss
    # loss = loss_function.calculate(output=activation2.output, y=y)

    # new part
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = loss_activation.forward(dense2.output, y)

    # Let's see output of the first few samples:
    print(loss_activation.output[:5])




    # Print loss value
    print('loss:', loss)



    # # Accuracy part
    # # Calculate values along second axis (axis of index 1)
    # predictions = np.argmax(activation2.output, axis=1)
    # # If targets are one-hot encoded - convert them
    # if len(y.shape) == 2:
    #     class_targets = np.argmax(y, axis=1)
    # # True evaluates to 1; False to 0
    # accuracy = np.mean(predictions == y)
    # print('accuracy:', accuracy)



    # new part

    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    # Print accuracy
    print('acc:', accuracy)

    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Print gradients
    print(dense1.dweights)
    print(dense1.dbiases)
    print(dense2.dweights)
    print(dense2.dbiases)




    # ---------------------------------------
    # Backpropagation testing p. 234

    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])

    class_targets = np.array([0, 1, 1])

    softmax_loss = af.Activation_Softmax_Loss_CategoricalCrossentropy()
    softmax_loss.backward(softmax_outputs, class_targets)
    dvalues1 = softmax_loss.dinputs

    activation = af.Activation_Softmax()
    activation.output = softmax_outputs
    loss = ccel.Loss_CategoricalCrossentropy()
    loss.backward(softmax_outputs, class_targets)
    activation.backward(loss.dinputs)
    dvalues2 = activation.dinputs

    print('Gradients: combined loss and activation:')
    print(dvalues1)
    print('Gradients: separate loss and activation:')
    print(dvalues2)

    # ---------------------------------------
    # Backpropagation testing p. 235

    from timeit import timeit

    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])
    class_targets = np.array([0, 1, 1])


    def f1():
        softmax_loss = af.Activation_Softmax_Loss_CategoricalCrossentropy()
        softmax_loss.backward(softmax_outputs, class_targets)
        dvalues1 = softmax_loss.dinputs


    def f2():
        activation = af.Activation_Softmax()
        activation.output = softmax_outputs
        loss = ccel.Loss_CategoricalCrossentropy()
        loss.backward(softmax_outputs, class_targets)
        activation.backward(loss.dinputs)
        dvalues2 = activation.dinputs


    t1 = timeit(lambda: f1(), number=10000)
    t2 = timeit(lambda: f2(), number=10000)
    print(t2 / t1)




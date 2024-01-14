import numpy as np


# first "dense layer" for first tests, replaced with layer_dense
class linear:

    # CLASS DATA

    # def __init__(self, neurons_number : int, input_size : int):
    #     self.weights = np.random.uniform(-1.0, 1.0, (neurons_number, input_size))
    #     self.biases = np.random.uniform(-1.0, 1.0, neurons_number)

    def __init__(self, weights : list, biases : list):
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        if self.biases.shape[0] != self.weights.shape[0]:
            raise Exception("Error! Biases size must be equal to the number of the neurons weights.\n")


    def forward(self, inputs : list) -> np:
        input_np = np.array(inputs)
        if input_np.shape[0] != self.weights.shape[0]:
            raise Exception("Error! Input array size must be equal to the number of the neurons weights.\n")
        return np.dot(input_np, self.weights.T) + self.biases
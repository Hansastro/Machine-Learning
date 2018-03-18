import numpy as np
import scipy.special

class neuralNetwork:
    '''
    Simple Neural network with three layer. Input, Hidden, Output
    '''
    def __init__(self, nbInputNodes, nbHiddenNodes, nbOutputNodes):
        # Weight table for links between input and hidden layer
        self.wih_ = np.random.normal(0.0, \
                                     pow(nbHiddenNodes, -0.5), \
                                         (nbHiddenNodes, nbInputNodes))

        # Weight table for links between hidden and output
        self.who_ = np.random.normal(0.0, \
                                    pow(nbOutputNodes, -0.5), \
                                    (nbOutputNodes, nbHiddenNodes))

        # Define a default learning rate
        self.learningRate_ = 0.01

        # Define an activation function
        # Here a sigmoid
        self.activationFunction_ = lambda x: scipy.special.expit(x)

    def predict(self, inputData):
        inputs = np.array(inputData, ndmin=2).T
        hiddenOutputs = np.dot(self.wih_, inputs)
        hiddenOutputs = self.activationFunction_(hiddenOutputs)

        outputs = np.dot(self.who_, hiddenOutputs)
        outputs = self.activationFunction_(outputs)

        return outputs

    def train(self, learningRate = 0.01):
        pass

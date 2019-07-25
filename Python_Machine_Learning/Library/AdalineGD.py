import numpy as np

class AdalineGD(object):
    '''
    Easy Perceptron with gradient descent+
    No hidden Layer

    Parameters:
        - eta: float
            Leanrning rate (between 0 and 1.0)
        - n_iter: int
            Number of iteration (epoch)
        - randomState: int
            seed for the random number generator (used to initialized the weight of the neural network)
    '''

    def __init__(self, eta=0.01, n_iter=50, randomState=1):
        self.eta = eta
        self.n_iter = n_iter
        self.randomState = randomState
        print("AdalineGD Initialization...")

    def fit(self, X, y):
        '''Train the neural network

        Parameters:
        X: Input Vector. (Array n_samples, n_features)
        y: target output. (Array n_samples)
        '''
        rgen = np.random.RandomState(self.randomState)
        # Generate a random value for all connexions plus the common one
        # Number of connexion + 1
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        '''Calculate the net input'''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        '''Compute inear activation'''
        return X

    def predict(self, X):
        '''Return the class label after unit step'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)




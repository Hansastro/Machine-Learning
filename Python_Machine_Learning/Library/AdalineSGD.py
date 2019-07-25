
import numpy as np

class AdalineSGD(object):
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

    def __init__(self, eta=0.01, n_iter=50, shuffle=True, randomState=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.randomState = randomState
        print("AdalineSGD Initialization...")

    def fit(self, X, y):
        '''Train the neural network

        Parameters:
        X: Input Vector. (Array n_samples, n_features)
        y: target output. (Array n_samples)
        '''
        self._initialize_weight(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
                cost = []
                for xi, target in zip(X, y):
                    cost.append(self._update_weights(xi, target))
                avg_cost = sum(cost) / len(y)
                self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        '''
        Fit trainings data without reinitializeing the weights
        '''
        if not self.w_initialized:
            self._initialize_weight(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weigths(X, y)
        return self

    def _shuffle(self, X, y):
        '''
        Shuffle training data
        '''
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weight(self, m):
        '''
        Initialize weights to small random numbers
        :param m: size a the weight table to initialize
        :return:
            None
        '''
        self.rgen = np.random.RandomState(self.randomState)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        '''
        Apply the Adaline learning rule to update the weights
        '''
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        '''Calculate the net input'''
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        '''Compute inear activation'''
        return X

    def predict(self, X):
        '''Return the class label after unit step'''
        return np.where(self.net_input(X) >= 0.0, 1, -1)




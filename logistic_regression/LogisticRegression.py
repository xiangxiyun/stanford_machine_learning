import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression:

    def _sigmoid(self, z):
        '''
        perform the sigmoid function on every element
        :param z: size: m*1
        :return: size: m*1
        '''
        new_z = np.array(z)
        for i in range(z.shape[0]):
            new_z[i, 0] = 1/(1 + np.exp(0-z[i,0]))
        return new_z

    def _hypothesis_function(self, X, theta):
        '''
        h = sigmoid(X * theta)
        :param X:  input features, size: m*(n+1)
        :param theta: size: (n+1)*1
        :return: numpay array, size: m*1
        '''
        return self._sigmoid(np.dot(X, theta))

    def cost_function(self, X, y, theta):
        m = X.shape[0]
        h = self._hypothesis_function(X, theta)

        J = (1/m)*np.sum( (0-y) * np.log(h) - (1-y)*np.log(1-h))

        return J

    def gradient_descent(self, X, y, theta):
        m = X.shape[0]
        h = self._hypothesis_function(X, theta)
        return (1/m)* (X.T *(h-y))

    def predict(self):
        # TODO:
        pass

    def cost_function_reg(self):
        # TODO
        pass


    def training(self):
        pass
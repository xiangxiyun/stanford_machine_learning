import numpy as np
from scipy.optimize import fmin_bfgs
from matplotlib import pyplot as plt


class LogisticRegression:

    def _sigmoid(self, z):
        '''
        perform the sigmoid function on every element
        :param z: size: m*1
        :return: size: m*1
        '''

        return 1/(1+np.exp(0-z))

    def _hypothesis_function(self, X, theta):
        '''
        h = sigmoid(X * theta)
        :param X:  input features, size: m*(n+1)
        :param theta: size: (n+1)*1
        :return: numpay array, size: m*1
        '''
        return self._sigmoid(np.dot(X, theta))


    def cost_function(self, X, y, theta):
        '''

        :param X:
        :param y:
        :param theta:
        :return:
        '''
        m = X.shape[0]
        h = self._hypothesis_function(X, theta)

        J = (1/m)*np.sum( (0-y) * np.log(h) - (1-y)*np.log(1-h) )

        return J

    def _derivation_function(self, X, y, theta, h):
        m = X.shape[0]
        h = self._hypothesis_function(X, theta)
        return (1/m)* np.dot(X.T ,(h-y))

    def _calculate_mean_std(self, X):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)

    def _feature_normalize(self, X, m):

        for i in range(m):
            X[i, :] = (X[i, :] - self.X_mean) / self.X_std

        return X

    def gradient_descent(self, X, y, learning_rate, theta, iteration):
        '''

        :param X:
        :param y:
        :param theta:
        :return:
        '''
        m = X.shape[0]
        self.theta = theta
        self.cost_history[0, 0] = self.cost_function(X, y, self.theta)


        for i in range(iteration):
            # hypothesis function
            h = self._hypothesis_function(X, self.theta)

            # Gradient
            der_J = self._derivation_function(X, y, self.theta, h)

            # Update theta
            self.theta = self.theta - learning_rate * der_J


            self.cost_history[0, i] = self.cost_function(X, y, self.theta)

        return self.theta


    def newton_conjugate_gradient(self, X, y, theta, iteration):
        '''
        Using Newton's Conjugate Gradient method to minimize cost function.
        Utilizing Hessian matrix(second-order partial derivatives) of cost function.
        :param X:
        :param y:
        :param theta:
        :param iteration:
        :return:
        '''

        m = X.shape[0]
        self.theta = theta
        self.cost_history[0,0] = self.cost_function(X, y, self.theta)


        for i in range(iteration):

            h = self._hypothesis_function(X, self.theta)

            # Hessian Matrix
            H = (1/m)* ( np.dot(h.T, (1-h)) * np.dot(X.T, X) )

            # Gradient
            der_J = self._derivation_function(X, y, self.theta, h)

            # Update theta
            self.theta = self.theta - np.dot(np.linalg.pinv(H), der_J)

            self.cost_history[0, i] = self.cost_function(X, y, self.theta)

        return self.theta


    def training(self, X, y, iteration = 400, method = 'G', learning_rate = 0.01, normalization = False):

        m = X.shape[0] # total number of training data
        n = X.shape[1] # total number of features

        self.X = np.array(X)
        self.y = np.array(y)

        theta = np.zeros((n+1, 1))

        self.cost_history = np.zeros((1,iteration))

        if method == 'N': # Newton' method
            self.X = np.concatenate((np.ones((m, 1), dtype=float), self.X),
                                    axis=1)  # create corresponding x0 for theta0
            return self.newton_conjugate_gradient(self.X, self.y, theta, iteration)
        elif method == 'G': # gradient descent
            if normalization:
                self._calculate_mean_std(self.X)
                self.X = self._feature_normalize(self.X, m)
            self.X = np.concatenate((np.ones((m, 1), dtype=float), self.X),
                                    axis=1)  # create corresponding x0 for theta0
            return self.gradient_descent(self.X, self.y, learning_rate, theta, iteration)


    def predict(self, X, theta):
        new_X = np.array(X)
        m = X.shape[0]

        new_X = np.concatenate((np.ones((m, 1), dtype=float), new_X), axis=1)

        return int(self._hypothesis_function(new_X, theta)>=0.5)


    def cost_function_reg(self):
        # TODO
        pass

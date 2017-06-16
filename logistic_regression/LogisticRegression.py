import numpy as np
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

    def gradient_function(self, X, y, theta):
        '''

        :param X:
        :param y:
        :param theta:
        :return:
        '''
        m = X.shape[0]
        h = self._hypothesis_function(X, theta)
        return (1/m)* (X.T *(h-y))


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
            der_J = (1/m)* np.dot(X.T ,(h-y))

            # Update theta
            self.theta = self.theta - np.dot(np.linalg.pinv(H), der_J)

            self.cost_history[0, i] = self.cost_function(X, y, self.theta)

        return self.theta


    def training(self, X, y, iteration = 400, method = 'N'):

        m = X.shape[0] # total number of training data
        n = X.shape[1] # total number of features

        self.X = np.array(X)
        self.y = np.array(y)

        # if method == 'N':

        theta = np.zeros((n+1, 1))
        self.X = np.concatenate((np.ones((m, 1), dtype=float), self.X),
                                axis=1)  # create corresponding x0 for theta0

        self.cost_history = np.zeros((1,iteration))

        return self.newton_conjugate_gradient(self.X, self.y, theta, iteration)


    def predict(self, X, theta,):
        new_X = np.array(X)
        m = X.shape[0]

        new_X = np.concatenate((np.ones((m, 1), dtype=float), new_X), axis=1)

        return self._hypothesis_function(new_X, theta)


    def cost_function_reg(self):
        # TODO
        pass

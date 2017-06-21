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
        res = 1/(1+np.exp(0-z))
        idx = res == 1  # incase log(1) = 0
        res[idx] = .99
        return res

    def _hypothesis_function(self, X, theta):
        '''
        h = sigmoid(X * theta)
        :param X:  input features, size: m*(n+1)
        :param theta: size: (n+1)*1
        :return: numpy array, size: m*1
        '''
        return self._sigmoid(np.dot(X, theta))


    def _derivation_function(self, theta, X, y, _lambda = 0):
        '''

        :param X:
        :param y:
        :param theta:
        :param h:
        :return:
        '''
        m = X.shape[0]
        new_X = np.concatenate((np.ones((m, 1), dtype=float), X),
                axis=1)  # create corresponding x0 for theta0
        h = self._hypothesis_function(new_X, theta)

        grad = (1/m)* np.dot(new_X.T ,(h-np.ndarray.flatten(y)))

        grad[1:] += _lambda/m*theta[1:]

        return grad

    def _calculate_mean_std(self, X):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)

    def _feature_normalize(self, X, m):

        for i in range(m):
            X[i, :] = (X[i, :] - self.X_mean) / self.X_std

        return X

    def cost_function(self, theta, X, y, _lambda = 0):
        '''

        :param X: numpy array, size m*(n+1),
                  m is the number of training data,
                  n is the number of features.
        :param y: numpy array, size m*1.
        :param theta: numpy array, size (n+1)*1.
        :return: float, J(cost).

        '''
        m = X.shape[0]
        new_X = np.concatenate((np.ones((m, 1), dtype=float), X),
                        axis=1)  # create corresponding x0 for theta0

        h = self._hypothesis_function(new_X, theta)

        # J = (1/m)*np.sum( (0-y) * np.log(h) - (1-y)*np.log(1-h) )
        J = -1.0 / m * (np.dot(y.T, np.log(h))+ np.dot(1 - y.T, np.log(1 - h)))
        J += _lambda/(2*m)*np.sum(theta[1:]**2)

        return J


    def gradient_descent(self, X, y, learning_rate, theta, iteration):
        '''

        :param X:
        :param y:
        :param theta:
        :return:
        '''
        m = X.shape[0]
        self.theta = theta
        self.cost_history[0, 0] = self.cost_function(self.theta, X, y)

        new_X = np.concatenate((np.ones((m, 1), dtype=float), X),
            axis=1)  # create corresponding x0 for theta0


        for i in range(1, iteration):
            # hypothesis function
            h = self._hypothesis_function(new_X, self.theta)

            # Gradient
            # der_J = self._derivation_function(self.theta, X, y)
            # don't use self._derivation_function because the dimension
            # of y is customized for fmin_bfgs function
            der_J = (1/m)* np.dot(new_X.T ,(h-y))

            # Update theta
            self.theta = self.theta - learning_rate * der_J


            self.cost_history[0, i] = self.cost_function(self.theta, X, y)[0,0]

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
        self.cost_history[0,0] = self.cost_function(self.theta, X, y)

        new_X = np.concatenate((np.ones((m, 1), dtype=float), X),
            axis=1)  # create corresponding x0 for theta0

        for i in range(1,iteration):


            h = self._hypothesis_function(new_X, self.theta)

            # Hessian Matrix
            H = (1/m)* ( np.dot(h.T, (1-h))[0,0] * np.dot(new_X.T, new_X) )


            # Gradient
            # der_J = self._derivation_function(self.theta, X, y)
            # don't use self._derivation_function because the dimension
            # of y is customized for fmin_bfgs function
            der_J = (1/m)* np.dot(new_X.T ,(h-y))

            # Update theta
            a = np.linalg.pinv(H)
            self.theta = self.theta - np.dot(a , der_J)

            self.cost_history[0, i] = self.cost_function(self.theta, X, y)[0,0]


        return self.theta


    def using_fmin_bfgs(self,theta, iteration = 400, _lambda = 0):

        xopt = fmin_bfgs(self.cost_function, theta, \
                         fprime=self._derivation_function, \
                         args = (self.X, self.y, _lambda), maxiter=iteration)

        return xopt

    def training(self, X, y, iteration = 400, method = 'newton', learning_rate = 0.01, _lambda = 0):

        m = X.shape[0] # total number of training data
        n = X.shape[1] # total number of features

        self.X = np.array(X)
        self.y = np.array(y)
        self.method = method

        theta = np.zeros((n+1, 1))

        self.cost_history = np.zeros((1,iteration))

        if method == 'newton': # Newton's method
            return self.newton_conjugate_gradient(self.X, self.y, theta, iteration)

        elif method == 'gradient': # Gradient descent method

            # normalization
            self._calculate_mean_std(self.X)
            self.X = self._feature_normalize(self.X, m)

            return self.gradient_descent(self.X, self.y, learning_rate, theta, iteration)

        elif method == 'bfgs': # fmin_bfgs method
            return fmin_bfgs(self.cost_function, theta, \
                         fprime=self._derivation_function, \
                         args = (self.X, self.y, _lambda), maxiter=iteration)



    def predict(self, X, theta):
        m = X.shape[0]

        # if we do normalize training data
        # we need to do the same on testing data
        if self.method == 'gradient':
            X = self._feature_normalize(X, m)

        new_X = np.concatenate((np.ones((m, 1), dtype=float), X), axis=1)

        #return int(self._hypothesis_function(new_X, theta)>=0.5)
        return self._hypothesis_function(new_X, theta)

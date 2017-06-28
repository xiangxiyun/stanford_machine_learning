import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import fmin_cg

class MulticlassClassification:
    def __init__(self):
        pass

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

    def training(self, X, y, num_labels, _lambda, iteration):

        n = X.shape[1]
        theta = np.zeros((n+1, 1))

        all_theta = np.zeros((num_labels, n + 1)) #size: 10*401


        for i in range(num_labels):
            new_y = np.array(y == i, dtype=float)

            theta = fmin_cg(self.cost_function, theta, \
                         fprime=self._derivation_function, \
                         args = (X, new_y, _lambda), gtol=1e-05, maxiter=iteration)

            all_theta[i, :] = theta.T

        return all_theta


    def predictOneVsAll(self, X, all_theta):
        m = X.shape[0]
        new_X = np.concatenate((np.ones((m, 1), dtype=float), X),
                axis=1)  # create corresponding x0 for theta0

        res = self._sigmoid(np.dot(new_X, all_theta.T))
        p = np.argmax(res, axis = 1) #size: 5000*1

        return p[:, None]

    # debug
    def showResult(self, X, y, all_theta):
        m = X.shape[0]
        new_X = np.concatenate((np.ones((m, 1), dtype=float), X),
        axis=1)  # create corresponding x0 for theta0

        compute = self._sigmoid( np.dot(new_X, np.transpose(all_theta)) )
        p= np.argmax(compute, axis=1)[:, None]

        count = 0
        for k in range(m):
            if y[k] != p[k]:
                count+=1

        print('Total number of training data: ', m)
        print('Number of different pairs: ', count)





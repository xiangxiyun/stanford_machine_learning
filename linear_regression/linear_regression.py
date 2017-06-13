import numpy as np

class LinearRegression:

    def _hypothesis_function(self, X, theta):
        return np.dot(X, theta)  #h = X * theta  // result shape: m*1

    def cost_function(self, X, y, theta, m):
        h = self._hypothesis_function(X, theta)
        return (1/(2*m))*np.sum(np.power(h-y, 2))

    def gradient_descent(self, X, y, theta, learning_rate, iteration, m):

        self.theta = theta

        for i in range(iteration):
            # hypothesis function
            h = self._hypothesis_function(X, self.theta)
            print(h)

            # update theta
            self.theta = self.theta - learning_rate * (1/float(m)) * np.dot(X.T, (h-y))

            # cost function
            J = self.cost_function(X, y, self.theta, m)

            self.cost_history[0, i] = J

        return self.theta


    def training(self, X, y, learning_rate = 0.01, iteration = 400):

        m = X.shape[0] # total number of training data
        n = X.shape[1] # total number of features

        self.X = np.concatenate((np.ones((m,1), dtype= float),X), axis = 1) #create corresponding x0 for theta0
        self.y = y

        theta = np.zeros((n+1, 1))
        # print('theta shape:')
        # print(theta.shape)

        self.cost_history = np.zeros((1,iteration))

        # return theta
        return self.gradient_descent(self.X, self.y, theta, learning_rate, iteration, m)


    def predict(self, X, theta):
        return np.sum(self._hypothesis_function(X, theta))




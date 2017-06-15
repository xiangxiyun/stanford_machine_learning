import numpy as np

class LinearRegression:

    def _hypothesis_function(self, X, theta):
        return np.dot(X, theta)  #h = X * theta  // result shape: m*1

    def cost_function(self, X, y, theta, m):
        h = self._hypothesis_function(X, theta)
        return (1/(2*m))*np.sum(np.power(h-y, 2))

    def gradient_descent(self, X, y, theta, learning_rate, iteration, m):

        self.theta = theta
        self.cost_history[0,0] = self.cost_function(X, y, self.theta, m)

        for i in range(1,iteration):
            # hypothesis function
            h = self._hypothesis_function(X, self.theta)

            # update theta
            self.theta = self.theta - learning_rate * (1/float(m)) * np.dot(X.T, (h-y))

            # cost function
            J = self.cost_function(X, y, self.theta, m)

            self.cost_history[0, i] = J

        return self.theta

    def normal_equation(self, X, y):
        return np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)

    def _calculate_mean_std(self, X):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)

    def _feature_normalize(self, X, m):

        for i in range(m):
            X[i, :] = (X[i, :] - self.X_mean) / self.X_std

        return X


    def training(self, X, y, learning_rate = 0.01, iteration = 400, normalization = False, method = 'G'):


        m = X.shape[0] # total number of training data
        n = X.shape[1] # total number of features

        self.X = np.array(X)
        self.y = np.array(y)

        if method == 'N':
            self.X = np.concatenate((np.ones((m, 1), dtype=float), self.X),
                                    axis=1)  # create corresponding x0 for theta0
            return self.normal_equation(self.X, self.y)

        else:

            if normalization:
                self._calculate_mean_std(X)
                self.X = self._feature_normalize(self.X,m)

            self.X = np.concatenate((np.ones((m,1), dtype= float),self.X), axis = 1) #create corresponding x0 for theta0


            theta = np.zeros((n+1, 1))

            self.cost_history = np.zeros((1,iteration))

            # return theta
            return self.gradient_descent(self.X, self.y, theta, learning_rate, iteration, m)




    def predict(self, X, theta, normalization = False):
        new_X = np.array(X)
        m = X.shape[0]
        if normalization:
            new_X = self._feature_normalize(new_X, m)

        new_X = np.concatenate((np.ones((m, 1), dtype=float), new_X), axis=1)

        return self._hypothesis_function(new_X, theta)




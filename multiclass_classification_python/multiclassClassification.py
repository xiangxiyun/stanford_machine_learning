import numpy as np
from logistic_regression_python.logistic_regression import LogisticRegression

class MulticlassClassification:
    def __init__(self):
        pass

    def training(self, X, y, num_labels, _lambda):
        m = X.shape[0]
        n = X.shape[1]

        all_theta = np.zeros((num_labels, n + 1)) #size: 10*401
        LR = LogisticRegression()

        for i in range(num_labels):
            new_y = np.array(y == i, dtype=int)
            theta = LR.training(X, new_y, method = 'bfgs', _lambda= _lambda, iteration=1000)
            all_theta[i, :] = theta.T[:]

        return all_theta


    def predictOneVsAll(self, X, all_theta):
        m = X.shape[0]
        new_X = np.concatenate((np.ones((m, 1), dtype=float), X), axis=1) # size: 5000*401
        res = np.dot(new_X, all_theta.T) # size: 5000*10
        p = np.argmax(res, axis = 1).T #size: 5000*1
        return p




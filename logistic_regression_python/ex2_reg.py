import numpy as np
from matplotlib import pyplot as plt
from LogisticRegression import LogisticRegression


def read_data(filename):
    '''
    Read in training data
    :param filename: string
    :return: numpy array
    '''
    training_data_set = []
    with open(filename) as inputfile:
        for row in inputfile:
            single_data = [float(ele) for ele in row.split(',')]
            training_data_set.append(single_data)

    return np.array(training_data_set)

def map_features(x1, x2):
    '''
    map_feature(x1, x2) maps the two input features
    to quadratic features used in the regularization exercise.
    :param x1: numpy array, size m*1
    :param x2: numpy array, size m*1
    :return: numpy array, size m*28
    '''
    degree = 6
    out = np.ones(x1.shape)
    for i in range(1,degree+1):
        for j in range(i+1):
            hold = (x1**(i-j))*(x2**j)
            out = np.concatenate((out, hold), 1)

    return out





if __name__ == '__main__':

    input_filename = 'data/ex2data2.txt'
    training_data = read_data(input_filename)

    X = training_data[:, :-1]
    y = training_data[:, -1:]

    print('\nX shape:')
    print(X.shape)
    print('Y shape:')
    print(y.shape)

    # Plot training data
    neg = X[np.where(y == 0)[0], :]
    pos = X[np.where(y == 1)[0], :]

    plt.figure(1)
    plt.plot(pos[:, 0], pos[:, 1], 'g+', label = 'Admitted')
    plt.plot(neg[:, 0], neg[:, 1], 'yo', label = 'Not Admitted')

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Scatter plot of training data')

    # Begin training
    myLR = LogisticRegression()

    iteration = 6000
    learning_rate = 0.1
    normaliztion = False

    # 'N' represents Newton's method, 'G' represents gradient descent
    theta = myLR.training(map_features(X[:, 0:1], X[:,1:]), y, iteration, 'N', learning_rate, normalization=normaliztion, _lambda=1)

    print('Predict Result of [0, -0.8]:')
    new_X = map_features(np.array([[0]]),np.array([[-0.8]]))
    print(myLR.predict(new_X, theta))

    print('\n\nTheta:')
    print(theta.T)


    # Plot Decision boundary
    # def draw_boundary(classifier):
    #     dim = np.linspace(-1, 1.5, 1000)
    #     dx, dy = np.meshgrid(dim, dim)
    #     v = map_features(dx.flatten(), dy.flatten(), order=6)
    #     z = (np.dot(classifier.coef_, v) + classifier.intercept_).reshape(1000, 1000)
    #     CS = plt.contour(dx, dy, z, levels=[0], colors=['r'])
    #
    #
    # render_tests(data, accepted, rejected)
    # draw_boundary(classifier)
    plt.legend();


    plt.legend(loc = 0)

    # Plot cost history

    print('Cost:')
    print(myLR.cost_history.T[5900,:])
    plt.figure(2)
    plt.plot(np.arange(0, 2000, 1), myLR.cost_history.T[:2000,:], 'b-')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.title('Convergence of Newton Conjugate Gradient')

    plt.show()

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


def plot_data(X, y):
    '''
    Visualizing the training data
    :param X: Input features, numpy array
    :param y: Input classes, numpy array
    :return: None
    '''
    neg = X[np.where(y == 0)[0], :]
    pos = X[np.where(y == 1)[0], :]

    plt.figure()
    plt.plot(pos[:, 0], pos[:, 1], 'g+', label = 'Admitted')
    plt.plot(neg[:, 0], neg[:, 1], 'yo', label = 'Not Admitted')
    plt.legend(loc = 0)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()






if __name__ == '__main__':

    input_filename = 'data/ex2data1.txt'
    training_data = read_data(input_filename)

    X = training_data[:, :-1]
    y = training_data[:, -1:]

    # Plot trainin data
    plot_data(X, y)

    myLR = LogisticRegression()

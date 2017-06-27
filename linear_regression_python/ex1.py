import numpy as np
import matplotlib.pyplot as plt
from linear_regression_python.linear_regression import LinearRegression

def read_data(filename):
    '''
    read in training data
    :param filename: string
    :return: numpy array
    '''
    training_data_set = []
    with open(filename) as inputfile:
        for row in inputfile:
            single_data = [float(ele) for ele in row.split(',')]
            training_data_set.append(single_data)

    return np.array(training_data_set)



if __name__ == '__main__':

    # training data
    input_filename = 'data/ex1data1.txt'
    training_data = read_data(input_filename) # numpy array

    X = training_data[:, :-1]
    y = training_data[:, -1:]


    myLR = LinearRegression()

    print('\nX shape:')
    print(X.shape)
    print('Y shape:')
    print(y.shape)

    # plot training data
    plt.figure(1)
    plt.plot(X, y, 'g+')
    plt.title('Training Data and Linear Fit')


    iteration = 1500
    learning_rate = 0.01


    # First looking at the feature values.
    # When features differ by orders of magnitude,
    # first performing feature scaling.
    method = 'G'
    normalization = False
    theta = myLR.training(X, y, learning_rate, iteration, normalization=normalization, method = method)
    print('Theta:')
    print(theta.T)


    # plot fit line
    res = myLR.predict(X, theta, normalization=normalization)
    plt.plot(X, res, 'y-')


    # If method = 'G', gradient descent, plot cost history
    if method == 'G':
        plt.figure(2)
        plt.plot(np.arange(0, 30, 1), myLR.cost_history.T[:30,:], 'b-')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Cost')
        plt.title('Convergence of Gradient Descent')

    plt.show()

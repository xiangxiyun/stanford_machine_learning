import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    input_filename = 'data/ex1data2.txt'
    training_data = read_data(input_filename) # numpy array

    X = training_data[:, :-1]
    y = training_data[:, -1:]

    myLR = LinearRegression()


    print('\n\nX shape:')
    print(X.shape)
    print('Y shape:')
    print(y.shape)

    # Plot training data
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    x1s = X[:, 0:1]
    x2s = X[:, 1:2]
    ys = y

    scatter1 = ax.scatter(x1s, x2s, ys, c='b', marker='^', label = 'training data')

    ax.set_xlabel('x1(area)')
    ax.set_ylabel('x2(#bedroom)')
    ax.set_zlabel('y(price)')


    # Begin training data
    iteration = 1500
    learning_rate = 0.1



    method = 'G'    # Using gradient descent to calculate theta
    # First looking at the feature values.
    # When features differ by orders of magnitude,
    # first performing feature scaling.
    normalization = True

    theta = myLR.training(X, y, learning_rate, iteration, normalization=normalization, method = method)
    print('Theta:')
    print(theta.T)


    # Plot fit result
    res = myLR.predict(X, theta, normalization=normalization)
    scatter2 = ax.scatter(x1s, x2s, res, c='r', marker='o', label = 'fitting result')

    fig.legend([scatter1, scatter2], ['training data', 'fitting data'])


    # if method = 'G', Plot cost history
    if method == 'G':
        plt.figure(2)
        plt.plot(np.arange(0, 30, 1), myLR.cost_history.T[:30,:], 'b-')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Cost')
        plt.title('Convergence of Gradient Descent')

    plt.show()

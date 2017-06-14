import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

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

    #test cost function
    test_X = np.concatenate((np.ones((X.shape[0], 1), dtype=float), X), axis=1)
    test_theta = np.array([[-1], [2]])
    test_y = y
    test_m = test_X.shape[0]
    print(' ------- Test cost function ------- ')
    print('With theta equals: [-1 ; 2]')
    print('Expected cost value (approx) 54.24')
    print(myLR.cost_function(test_X, test_y, test_theta, test_m))
    print(' ---------------------------------- ')


    print('\n\nX shape:')
    print(X.shape)
    print('Y shape:')
    print(y.shape)

    #plot training data
    plt.figure(1)
    plt.subplot(211)

    plt.plot(X, y, 'g+')
    plt.title('Training Data and Linear Fit')

    iteration = 1500
    learning_rate = 0.01


    # First looking at the feature values.
    # When features differ by orders of magnitude,
    # first performing feature scaling.
    theta = myLR.training(X, y, learning_rate, iteration, normalization=False)
    print('Theta:')
    print(theta.T)


    #print fit line
    x_axis = X[:, -1:]
    a = np.concatenate((np.ones((x_axis.shape[0], 1), dtype=float), x_axis), axis=1)
    res = np.dot(a, myLR.theta)

    plt.plot(x_axis, res, 'y-')


    #print cost history
    plt.subplot(212)
    plt.plot(np.arange(0, 30, 1), myLR.cost_history.T[:30,:], 'b-')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.title('Convergence of Gradient Descent')

    plt.show()

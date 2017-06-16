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







if __name__ == '__main__':

    input_filename = 'data/ex2data1.txt'
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

    iteration = 1500
    learning_rate = 0.1
    normaliztion = False

    # 'N' represents Newton's method, 'G' represents gradient descent
    theta = myLR.training(X, y, iteration, 'N', learning_rate, normalization=normaliztion)

    print('Predict Result of [80, 80]:')
    print(myLR.predict(np.array([[80, 80]]), theta))

    print('\n\nTheta:')
    print(theta.T)


    # Plot Decision boundary
    plot_x = np.array([np.min(X[:,1])-2,  np.max(X[:,1])+2])
    plot_y = (-1 / theta[2]) * (theta[1] * plot_x + theta[0])

    if normaliztion:
        plot_x = np.array([myLR.X[20, 1] - 2, myLR.X[30, 1] + 2]) *myLR.X_std[0] + myLR.X_mean[0]
        plot_y = (-1/theta[2])*(theta[1]*np.array([myLR.X[20, 1] - 2, myLR.X[30, 1] + 2]) + theta[0])*myLR.X_std[1] + myLR.X_mean[1]

    plt.plot(plot_x, plot_y, 'r-', label='Decision Boundary')
    plt.legend(loc = 0)

    # Plot cost history
    plt.figure(2)
    plt.plot(np.arange(0, 1500, 1), myLR.cost_history.T[:1500,:], 'b-')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.title('Convergence of Newton Conjugate Gradient')

    plt.show()

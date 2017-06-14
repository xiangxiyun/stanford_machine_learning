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
    input_filename = 'data/ex1data2.txt'
    training_data = read_data(input_filename) # numpy array

    X = training_data[:, :-1]
    y = training_data[:, -1:]


    myLR = LinearRegression()


    print('\n\nX shape:')
    print(X.shape)
    print('Y shape:')
    print(y.shape)


    iteration = 1500
    learning_rate = 0.1


    # First looking at the feature values.
    # When features differ by orders of magnitude,
    # first performing feature scaling.
    theta = myLR.training(X, y, learning_rate, iteration, normalization=True)
    print('Theta:')
    print(theta.T)


    #print cost history

    plt.plot(np.arange(0, 30, 1), myLR.cost_history.T[:30,:], 'b-')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cost')
    plt.title('Convergence of Gradient Descent')

    plt.show()

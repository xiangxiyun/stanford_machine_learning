import scipy.io

from ..logistic_regression_python import LogisticRegression



if __name__ == '__main__':

    myLR = LogisticRegression()

    input_filename1 = 'data/ex3data1.mat'
    training_data = scipy.io.loadmat(input_filename1)

    X = training_data['X']
    y = training_data['y']

    input_filename2 = 'data/ex3weights.mat'
    weights = scipy.io.loadmat(input_filename2)

    Theta1 = weights['Theta1']
    Theta2 = weights['Theta2']

    pass

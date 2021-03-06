import scipy.io
import numpy as np
from multiclass_classification_python.multiclassClassification import MulticlassClassification



if __name__ == '__main__':



    input_filename1 = 'data/ex3data1.mat'
    training_data = scipy.io.loadmat(input_filename1)

    X = training_data['X'] #size: 5000*400
    y = training_data['y'] #size: 5000*1

    # y range from 1 to 10, but python's iteration begins from 0
    y[y == 10] = 0

    myMC = MulticlassClassification()
    _lambda = 0.1
    num_labels = 10
    all_theta = myMC.training(X, y, num_labels, _lambda, 50)

    pred = myMC.predictOneVsAll(X, all_theta)

    print('\nTraining Set Accuracy: ', np.mean(np.array(pred == y, dtype = float)) * 100, '%')

    myMC.showResult(X, y, all_theta)


    # input_filename2 = 'data/ex3weights.mat'
    # weights = scipy.io.loadmat(input_filename2)
    #
    # Theta1 = weights['Theta1'] #size: 25*401
    # Theta2 = weights['Theta2'] #size: 10*26




    pass

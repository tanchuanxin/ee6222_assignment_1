import numpy as np
import random
from function import RVFL_train_val
import h5py
from option import option as op
from tqdm import tqdm
import csv
import datetime as datetime


# results csv file to store everything that we have found. initialize file with header
results_filename = 'results_' + str(datetime.datetime.now().strftime("%y%m%d_%H%M%S")) + '.csv'
with open(results_filename, 'w', newline='') as csvfile:
    header = ['dataset_name', 'dataset_shape', 'ACC_CV', 'mean_ACC_CV', 'var_ACC_CV']

    writer = csv.writer(csvfile)
    writer.writerow(header)



# datasets have been chosen to be reasonably large, while working within the constraints of computing resources
dataset_names = ["abalone", "bank", "car", "dermatology", "echocardiogram", "glass", "seeds", "teaching", "titanic", "wine"]


# we will loop through all of our chosen datasets

for dataset_name in dataset_names:
    # load in the dataset file
    temp = h5py.File("UCI data python\\" + dataset_name + "_R.mat")
    data = np.array(temp['data']).T

    dataset_shape = data.shape
    print("\nprocessing for dataset ", dataset_name, "...")
    print("dataset shape: ", dataset_shape)

    # select the X features
    data = data[:, 1:]
    dataX = data[:, 0:-1]

    # do normalization for each feature
    dataX_mean = np.mean(dataX, axis=0)
    dataX_std = np.std(dataX, axis=0)
    dataX = (dataX - dataX_mean) / dataX_std

    # select the Y features
    dataY = data[:, -1]
    dataY = np.expand_dims(dataY, 1)

    # load in the train-test index file
    temp = h5py.File("UCI data python\\" + dataset_name + "_conxuntos.mat")
    index1 = np.array(temp['index1']).astype(np.int32) - 1
    index2 = np.array(temp['index2']).astype(np.int32) - 1
    index1 = np.squeeze(index1, axis=1)
    index2 = np.squeeze(index2, axis=1)

    # assign train-test split
    trainX = dataX[index1, :]
    trainY = dataY[index1, :]
    testX = dataX[index2, :]
    testY = dataY[index2, :]


    # container variables to hold optimal settings for N, C, Scale
    MAX_acc = np.zeros([6, 1])
    Best_N = np.zeros([6, 1]).astype(np.int32) # Number of neurons in hidden layer
    Best_C = np.zeros([6, 1]) # Regularization parameter strength (lambda)
    Best_S = np.zeros([6, 1]) # Linear scale of random variables before feeding into non-linear activation function
    S = np.linspace(-5, 5, 21)

    # create six models to answer the three questions
    option1 = op() # for q1
    option2 = op() # for q2
    option3 = op() # for q3
    option4 = op() # for q4
    option5 = op() # for q5
    option6 = op() # for q6

    # experiment with different scale values to determine optimal N, C, Scale
    for s in tqdm(range(0, S.size)):
        # experiment with different number of hidden neurons
        for N in range(3, 204, 20):
            # experiment with different regularization parameters
            for C in range(-5, 15):
                Scale = np.power(2, S[s])

                # Question 1. Effect of direct links from the input layer to the output layer (i.e. with and without)
                # 0 for no link, 1 for link
                option1.N = N
                option1.C = 2 ** C
                option1.Scale = Scale
                option1.Scalemode = 3
                option1.link = 0

                option2.N = N
                option2.C = 2 ** C
                option2.Scale = Scale
                option2.Scalemode = 3
                option2.link = 1

                # Question 2. Performance comparisons of 2 activation functions: one from “relu, sigmoid, radbas, sine” and one from “hardlim, tribas”
                option3.N = N
                option3.C = 2 ** C
                option3.Scale = Scale
                option3.ActivationFunction = "radbas"
                option3.Scalemode = 3
                option3.link = 1

                option4.N = N
                option4.C = 2 ** C
                option4.Scale = Scale
                option4.ActivationFunction = "hardlim"
                option4.Scalemode = 3
                option4.link = 1

                # Question 3. Performance of Moore-Penrose pseudoinverse and ridge regression (or regularized least square solutions) for the computation of the output weights
                # 1 for regularized least square, 2 for moore-penrose pseudoinverse
                option5.N = N
                option5.C = 2 ** C
                option5.Scale = Scale
                option5.ActivationFunction = "radbas"
                option5.Scalemode = 3
                option5.link = 1
                option5.mode = 1

                option6.N = N
                option6.C = 2 ** C
                option6.Scale = Scale
                option6.ActivationFunction = "radbas"
                option6.Scalemode = 3
                option6.link = 1
                option6.mode = 2

                # train on all six models and get training train/test results
                train_accuracy1, test_accuracy1 = RVFL_train_val(trainX, trainY, testX, testY, option1)
                train_accuracy2, test_accuracy2 = RVFL_train_val(trainX, trainY, testX, testY, option2)
                train_accuracy3, test_accuracy3 = RVFL_train_val(trainX, trainY, testX, testY, option3)
                train_accuracy4, test_accuracy4 = RVFL_train_val(trainX, trainY, testX, testY, option4)
                train_accuracy5, test_accuracy5 = RVFL_train_val(trainX, trainY, testX, testY, option5)
                train_accuracy6, test_accuracy6 = RVFL_train_val(trainX, trainY, testX, testY, option6)

                # only keep the best scores
                if test_accuracy1 > MAX_acc[
                    0]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                    MAX_acc[0] = test_accuracy1
                    Best_N[0] = N
                    Best_C[0] = C
                    Best_S[0] = Scale

                if test_accuracy2 > MAX_acc[
                    1]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                    MAX_acc[1] = test_accuracy2
                    Best_N[1] = N
                    Best_C[1] = C
                    Best_S[1] = Scale

                if test_accuracy3 > MAX_acc[
                    2]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                    MAX_acc[2] = test_accuracy3
                    Best_N[2] = N
                    Best_C[2] = C
                    Best_S[2] = Scale

                if test_accuracy4 > MAX_acc[
                    3]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                    MAX_acc[3] = test_accuracy4
                    Best_N[3] = N
                    Best_C[3] = C
                    Best_S[3] = Scale

                if test_accuracy5 > MAX_acc[
                    4]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                    MAX_acc[4] = test_accuracy5
                    Best_N[4] = N
                    Best_C[4] = C
                    Best_S[4] = Scale

                if test_accuracy6 > MAX_acc[
                    5]:  # parameter tuning: we prefer the parameter which lead to better accuracy on the test data
                    MAX_acc[5] = test_accuracy6
                    Best_N[5] = N
                    Best_C[5] = C
                    Best_S[5] = Scale

    # load in the cross-validation index file
    temp = h5py.File("UCI data python\\" + dataset_name + "_conxuntos_kfold.mat")
    index = []

    # read out the eight sets of indexes
    for i in range(8):
        index_temp = np.array([temp[element[i]][:] for element in temp['index']]).astype(np.int32) - 1
        index_temp = np.squeeze(index_temp, axis=0)
        index_temp = np.squeeze(index_temp, axis=1)
        index.append(index_temp)

    # container variable for cross-validation
    ACC_CV = np.zeros([6, 4])

    # by now, we would have found the best test accuracy, and best N, C, Scale, we will perform cross-validation
    # use the best settings found
    for i in tqdm(range(4)):
        trainX = dataX[index[2 * i], :]
        trainY = dataY[index[2 * i], :]
        testX = dataX[index[2 * i + 1], :]
        testY = dataY[index[2 * i + 1], :]

        # Question 1. Effect of direct links from the input layer to the output layer (i.e. with and without)
        # 0 for no link, 1 for link
        option1.N = Best_N[0, 0]
        option1.C = 2 ** Best_C[0, 0]
        option1.Scale = Best_S[0, 0]
        option1.Scalemode = 3
        option1.link = 0

        option2.N = Best_N[1, 0]
        option2.C = 2 ** Best_C[1, 0]
        option2.Scale = Best_S[1, 0]
        option2.Scalemode = 3
        option2.link = 1

        # Question 2. Performance comparisons of 2 activation functions: one from “relu, sigmoid, radbas, sine” and one from “hardlim, tribas”
        option3.N = Best_N[2, 0]
        option3.C = 2 ** Best_C[2, 0]
        option3.Scale = Best_S[2, 0]
        option3.ActivationFunction = "radbas"
        option3.Scalemode = 3
        option3.link = 1

        option4.N = Best_N[3, 0]
        option4.C = 2 ** Best_C[3, 0]
        option4.Scale = Best_S[3, 0]
        option4.ActivationFunction = "hardlim"
        option4.Scalemode = 3
        option4.link = 1

        # Question 3. Performance of Moore-Penrose pseudoinverse and ridge regression (or regularized least square solutions) for the computation of the output weights
        # 1 for regularized least square, 2 for moore-penrose pseudoinverse
        option5.N = Best_N[4, 0]
        option5.C = 2 ** Best_C[4, 0]
        option5.Scale = Best_S[4, 0]
        option5.ActivationFunction = "radbas"
        option5.Scalemode = 3
        option5.link = 1
        option5.mode = 1

        option6.N = Best_N[5, 0]
        option6.C = 2 ** Best_C[5, 0]
        option6.Scale = Best_S[5, 0]
        option6.ActivationFunction = "radbas"
        option6.Scalemode = 3
        option6.link = 1
        option6.mode = 2

        # train on all six models and get training train/test results
        train_accuracy1, ACC_CV[0, i] = RVFL_train_val(trainX, trainY, testX, testY, option1)
        train_accuracy2, ACC_CV[1, i] = RVFL_train_val(trainX, trainY, testX, testY, option2)
        train_accuracy3, ACC_CV[2, i] = RVFL_train_val(trainX, trainY, testX, testY, option3)
        train_accuracy4, ACC_CV[3, i] = RVFL_train_val(trainX, trainY, testX, testY, option4)
        train_accuracy5, ACC_CV[4, i] = RVFL_train_val(trainX, trainY, testX, testY, option5)
        train_accuracy6, ACC_CV[5, i] = RVFL_train_val(trainX, trainY, testX, testY, option6)

    print("\ncompleted cross-validation...")
    print("ACC_CV:", ACC_CV)

    mean_ACC_CV = np.mean(ACC_CV, axis=1)
    print("Mean:", mean_ACC_CV)

    var_ACC_CV = np.var(ACC_CV, axis=1)
    print("Variance:", var_ACC_CV)


    # write our results into a file to be analyzed 
    with open(results_filename, 'a', newline='') as csvfile:
        for i in range(6):
            row = [dataset_name, dataset_shape, ACC_CV[i], mean_ACC_CV[i], var_ACC_CV[i]]
            row = [str(item) for item in row]

            writer = csv.writer(csvfile)
            writer.writerow(row)



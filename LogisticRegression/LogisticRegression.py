'''Implementation of Logistic Regression in python'''

import numpy as np
from math import exp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def errorfunction_gradient(weight, X, y):
    # X(N * d+1) weight(d+1 * 1)
    temp = [0]*len(X[0])  # temporary variable
    for i in range(len(X)):
        temp += (y[i]/(1+exp(y[i] * np.dot(weight, X[i]))))*X[i]  # formula of gradient
    temp = np.array(temp)  # convert type
    E_grad = temp / len(X)  # gradient
    E_grad_len = np.linalg.norm(E_grad)  # compute gradient vector length
    E_grad = E_grad/E_grad_len  # convert gradient vector to unit vector
    return E_grad  # return direction vector of gradient


def logisticRegression(dataX, y):  # logistic regression with gradient descent optimization
    alpha = 0.01  # learning rate
    maxCycle = 7000  # time of iteration
    X = np.hstack([np.ones((len(dataX), 1)), dataX])
    weight = np.random.random(len(dataX[0])+1)  # initialize weight
    print('Initial random weight:', weight)
    for k in range(maxCycle):
        dir = errorfunction_gradient(weight, X, y)
        weight = weight + alpha * dir  # update weight
    return weight


def sigmoid(x):  # sigmoid function
    return exp(x)/(1+exp(x))


def predict(model, X_predict):  # model is the weights of logistic regression
    y_predict = []  # initialize the prediction
    for d in range(len(X_predict)):
        temp = X_predict[d]
        temp.insert(0, 1)
        temp = np.dot(model, temp)
        score = sigmoid(temp)
        # tell the classification according to the score
        if score >= 1-score:
            pre = 1
        else:
            pre = -1
        y_predict.append(pre)
    return y_predict


def plot(X, Y, W):
    # Plot data into 3D space for showing classification.
    # Plot the 3D plane that segments the datas.
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # h = .02  # step size in the mesh

    for i in range(len(X)):
        x = X[i][1]
        y = X[i][2]
        z = X[i][3]
        if Y[i] == 1:
            _c_p = ax.scatter(x, y, z, c='b', marker='.')
        else:
            _r_p = ax.scatter(x, y, z, c='y', marker='.')

    ax.legend ([_c_p, _r_p], ['Label data = 1','Label data = -1'])
    xx, yy = np.meshgrid(np.arange(-0.2, 1.2, 0.02), np.arange(-0.2, 1.2, 0.02))
    zz = -(W[0] + W[1] * xx + W[2] * yy) / W[3]
    # Plot the segmentation plane
    ax.plot_surface(xx, yy, zz, color='b', alpha=0.3)
    plt.show()


# main
with open('classification.txt', 'r') as data:
    # extracting data
    d = data.read().splitlines()
    data_x, data_y = [], []
    for i in range(len(d)):
        data_x.append(d[i].split(',')[:3])
        for p in data_x:  # format data points
            for q in range(len(p)):
                p[q] = float(p[q])
        data_y.append(d[i].split(',')[-1])
        for j in range(len(data_y)):  # format data type
            data_y[j] = int(data_y[j])

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=0)

    # logistic regression
    weights = logisticRegression(X_train, y_train)
    print('According to training set, the optimized weight is: ', weights)

    # accuracies
    # compute train accuracy
    y_p_train = predict(weights, X_train)  # prediction result
    c_train = 0  # counter
    for i in range(len(y_p_train)):
        if y_p_train[i] == y_train[i]:
            c_train += 1
    print('Training Data Prediction Accuracy: %.2f' % (c_train / len(y_train)))
    # compute test accuracy
    y_p_test = predict(weights, X_test)  # prediction result
    c_test = 0  # counter
    for i in range(len(y_p_test)):
        if y_p_test[i] == y_test[i]:
            c_test += 1
    print('Test Data Prediction Accuracy: %.2f' % (c_test / len(y_test)))

    # plot the points and the plane in 3d space
    if len(data_x[0])-1 <= 3:
        plot(X_test, y_test, weights)

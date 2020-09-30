'''Implementation of Linear Regression in python'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def linearReressionModle(dataX, dataY):
    D = np.array(dataX).T
    Y = np.array(dataY).T
    W_opt = np.dot(np.dot(np.linalg.inv(np.dot(D, D.T)), D), Y)
    return W_opt


def predict(weight, X_toPredict):
    for i in range(len(X_toPredict)):
        X_toPredict[i].insert(0, 1)
    prediction = []
    for X in X_toPredict:
        prediction.append(np.dot(weight, X))
    return prediction


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
        z = Y[i]
        _c_p = ax.scatter(x, y, z, c='y', marker='.')

    #ax.legend ([_c_p, _r_p], ['Label data = 1','Label data = -1'])
    xx, yy = np.meshgrid(np.arange(-0.2, 1.2, 0.02), np.arange(-0.2, 1.2, 0.02))
    zz = W[0] + W[1] * xx + W[2] * yy
    # Plot the segmentation plane
    ax.plot_surface(xx, yy, zz, color='b', alpha=0.3)
    plt.show()


def main():
    with open('linear-regression.txt', 'r') as data:
        # preprocessing data
        d = data.read().splitlines()
        data_x, data_y = [], []
        for i in range(len(d)):
            data_x.append([1])
            data_x[-1].extend(d[i].split(',')[:2])
            for p in data_x:  # format data points
                for q in range(len(p)):
                    p[q] = float(p[q])
            data_y.append(d[i].split(',')[-1])
            for j in range(len(data_y)):  # format data type
                data_y[j] = float(data_y[j])
        # linear regression
        weight = linearReressionModle(data_x, data_y)
        print('The optimized weights are: ', weight)
        print('The intercept is:', weight[0])
        # plot the result
        plot(data_x, data_y, weight)


if __name__ == '__main__':
    main()

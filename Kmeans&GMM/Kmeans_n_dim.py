'''This is for implementing K-means algorithm with n-dimensional data.'''

import random
import matplotlib.pyplot as plt


def read_data():  # to read the given data
    data = []
    with open('clusters.txt', 'r') as f:
        for line in f:
            line = line[:-1]  # remove '\n'
            line = line.split(',')
            # convert data to float and initialize the nearest centroid index with -1
            data.append([float(i) for i in line])
    return data


def dim(data):  # detect dimension of the data points
    return len(data[0])


def distance(a, b):  # compute the squared distance of two points a, b
    return sum(map(lambda ai, bi: (ai-bi)**2, a, b))


def rangeCord(data):  # find ceiling and floor of each coordinate
    rangelist = []  # list to store range for each dimension
    for i in range(dim(data)):
        temp = []
        for l in data:
            temp.append(l[i])
        rangelist.append([min(temp), max(temp)])
    return rangelist


def centroid(rangelist, k):  # initialize k centroids within the range of coordinates randomly
    centroids = []
    for i in range(k):
        centroids.append([])
        for r in rangelist:
            centroids[-1].append(random.uniform(r[0], r[1]))  # random centroid coordinate is generated within the range
    return centroids


def allocateCentroid(point, centroids):  # to allocate the nearest centroid to each data point
    min_index = 0  # initialize the centroid index
    min_dist = distance(centroids[0], point)
    for i in range(1, len(centroids)):  # travel each centroid to find nearest one to current point
        tmp = distance(centroids[i], point)
        if (tmp < min_dist):
            min_dist = tmp
            min_index = i
    return min_index  # return the index of centroid


def alloList(data, centroids):  # to generate a list of centroid indexes to which the data points are allocated
    cenList = []
    for p in data:
        cenList.append(allocateCentroid(p, centroids))
    return cenList


def dataCen(data):  # compute the center of all data points
    n = dim(data)
    dcen = [0]*n  # initialize the center with 0 in each coordinate
    for i in range(len(data)):
        for j in range(n):
            dcen[j] += data[i][j]
    dcen = [x/len(data) for x in dcen]
    return dcen


def farpoint(data):  # find the farthest point from the center in case the number of clusters decrease in iteration
    dcen = dataCen(data)
    farthest = data[0]  # initialize the farthest point with the first point
    for p in data:
        if distance(p, dcen) > distance(farthest, dcen):
            farthest = p
    return data.index(farthest)  # return the index of the farthest point in data


def newCentroids(data, alloList, k):
    new_centroids = []  # initialize the list to store new centroids
    d = dim(data)
    tempCo = [0]*d  # a temporary value to store the coordinates of each new centroid
    for i in range(k):
        n = 0
        for p in range(len(alloList)):
            if alloList[p] == i:  # travel all points that allocated to a current centroid
                for j in range(d):
                    tempCo[j] += data[p][j]
                n += 1
        if n == 0:
            # allocate the farthest point as a centroid if the number of centroids decrease
            new_centroids.append(data[farpoint(read_data())])
        else:
            new_centroids.append([x / n for x in tempCo])
        tempCo = [0]*d
    return new_centroids


def k_means(data, k):  # function to implement EM in recursion
    center = centroid(rangeCord(data), k)
    oldCen = []
    allo = []
    i = 0  # counter
    while center != oldCen and i < 10000:  # stop when convergence happen or reaching max ireration
        allo = alloList(data, center)
        oldCen = center
        center = newCentroids(data, allo, k)
        i += 1
        if i == 10000:
            print('Reach the max iteration time.')
    print('Iteration time is: ', i)
    print('The allocated centroid index of each pointe is: ', allo)
    print('The centroids are: ', center)
    plotClusters(data, allo, center, k)  # try to plot the result from the algorithm
    return allo, center


def plotClusters(data, allo, centroids, k): # plot the clustered points in one canvas
    if len(data[0]) > 2:
        print('Cannot print high dimensional data.')
        return 0
    elif len(data[0]) == 2:
        color = 'ko'  # default color
        for i in range(k):
            if i == 0:  # change color when plotting different clusters
                color = 'bo'
            elif i == 1:
                color = 'go'
            elif i == 2:
                color = 'ro'
            elif i == 3:
                color = 'co'
            elif i == 4:
                color = 'mo'
            elif i == 5:
                color = 'yo'
            else:
                # remind users that the color will not change when there are too many clusters
                print('Cannot print such many clusters.')
            plotx = []
            ploty = []
            for j in range(len(allo)):
                if allo[j] == i:
                    plotx.append(data[j][0])
                    ploty.append(data[j][1])
            plt.plot(plotx, ploty, color)
        # plot centroids
        cx, cy = [], []
        for m in range(k):
            cx.append(centroids[m][0])
            cy.append(centroids[m][1])
        plt.plot(cx, cy, 'ko')
        plt.show()
    return 0


def main():  # Please change parameters of data and k here if needed.
    k = 3  # according to the assignment handout, k is set to 3
    data = read_data()
    k_means(data, k)
    return 0


if __name__ == '__main__':
    main()

'''Implementation of FastMap in python'''

#importing modules
import random
import matplotlib.pyplot as plt


def distDL(string1, string2):  # return Damerau–Levenshtein distance of two strings
    d = {}
    len1 = len(string1)
    len2 = len(string2)
    for i in range(-1,len1):
        d[(i,-1)] = i+1
    for j in range(-1,len2):
        d[(-1, j)] = j+1
    # dynamic programming to generate Damerau–Levenshtein distance
    for i in range(len1):
        for j in range(len2):
            if string1[i] == string2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(d[(i-1, j)] + 1,  # delete
                           d[(i, j-1)] + 1,  # insert
                           d[(i-1, j-1)] + cost)  # substitute
            if i and j and string1[i] == string2[j-1] and string1[i-1] == string2[j]:
                # consider transposition only when i,j>0
                d[(i,j)] = min(d[(i, j)], d[i-2,j-2] + cost)  # transpose adjacent letters
    return d[(len1-1, len2-1)]


def fastMap(data, k):  # input a list of data objects and specified dimension k
    # initialization
    fm = dict()
    for o in data:
        fm[o] = [0]
    c = 1  # counter of dimension starting from 1

    # compute coordinates dimension by dimension
    while c <= k:
        # find the farthest object pair
        # initialization
        temp = random.randint(0, len(data) - 1)  # pick a random object as start of search
        s = temp  # index of object as start object
        e = -1  # index of object as end object which is the farthest one from NO.s
        d = 0  # to store the squared distance between s and e
        q = 0  # counter
        while e != temp:  # normally recursion can end within 5 times
            if q != 0:
                temp = s
                s = e
            for i in range(len(data)):  # travel all objects to find farthest one from current NO.s object
                sqdistance = distDL(data[s], data[i]) ** 2  # squared distance base on original distance function
                for j in range(c):
                    # squared distance by new function; equal to original one when c = 1.
                    sqdistance = sqdistance - ((fm[data[s]][j] - fm[data[i]][j]) ** 2)
                if sqdistance >= d:  # update e and d
                    e = i
                    d = sqdistance
            q += 1
        # obtain the farthest objects
        a = data[s]
        b = data[e]
        # output the farthest object pair indeces.
        print('Farthest object pair consists of: #', s+1, a, 'and #', e+1, b)

        # compute NO.c coordinates of each object
        for o in data:
            # use distDL() as original distance function
            squareDist_ab = distDL(a, b) ** 2
            squareDist_ao = distDL(a, o) ** 2
            squareDist_bo = distDL(b, o) ** 2
            # use new distance function
            for i in range(c):
                squareDist_ab = squareDist_ab - ((fm[a][i] - fm[b][i]) ** 2)
                squareDist_ao = squareDist_ao - ((fm[a][i] - fm[o][i]) ** 2)
                squareDist_bo = squareDist_bo - ((fm[b][i] - fm[o][i]) ** 2)
            coordinate = (squareDist_ao + squareDist_ab - squareDist_bo)/(2 * (squareDist_ab ** 0.5))
            fm[o].append(coordinate)
        c += 1

    # format coordinates
    for w in fm.keys():
        fm[w] = fm[w][1:]
    print(fm)
    if k == 2:  # plot the points only if dimension equals to 2
        plotfm(fm)
    return fm


def plotfm(fmDict):
    # collect coordinates respectively
    xl = []
    yl = []
    for o in fmDict.keys():
        xl.append(fmDict[o][0])
        yl.append(fmDict[o][1])
        plt.annotate(o,  # the object name
                     (fmDict[o][0], fmDict[o][1]),  # point to represent the object
                     textcoords="offset points",
                     xytext=(0, 10),  # distance from label to point
                     ha='left')
    plt.scatter(xl, yl)
    plt.show()


def main():
    dimension = 2
    with open('fastmap-wordlist.txt', 'r') as f:
        l = f.readlines()
        for n in range(len(l)):
            l[n] = l[n].strip('\n')
        fastMap(l, dimension)
    return 0


if __name__ == '__main__':
    main()

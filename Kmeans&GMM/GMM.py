'''Implementation of GMM using EM in python'''

#importing modules
import random
import numpy as np
import math

DEBUG = 0

#read_data function to read data and generating data from text file
def read_data(filename):
    data = np.genfromtxt(filename, delimiter=',')
    return data

# dim function for detecting dimensions of the data points 
def dim(data):  # detect dimension of the data points
    return data.shape[1]

# initialize_data function is used for initializing ric based on random selection of ric.
def initialize_data_ric(data):
    datalist=[]
    for i in range(len(data)):
        dummy=[]
        dummy.append(data[i][0])
        dummy.append(data[i][1])
        val=random.choice(range(1,4)) # to choose value of ric randomly and assign to each data points
        for j in range(1,4):
            if j==val:
                dummy.append(1)
            else:
                dummy.append(0)
        datalist.append(dummy)
    return datalist

# calculating_covariance for calculation of covariance matrix which is of d*d dimensions
def calculating_covariance(riclist):
    colist = []
    for var in riclist:
        list1 = []
        list1.append(var[0])
        list1.append(var[1])
        colist.append(list1)
    x = np.array(colist).T
    covar = np.cov(x)
    return (covar)

# calculating_mean function for calculating mean of data points which is of (1*d) dimensions
def calculating_mean(riclist):
    meanlist = []
    for var in riclist:
        tmplist = []
        tmplist.append(var[0])
        tmplist.append(var[1])
        meanlist.append(tmplist)
    a = np.array(meanlist)
    var = np.mean(a, axis=0)
    return (var)

# initial_setup method is used for calculating initial mean(centroid), co-variance, initial_centroids, amplitude
def initial_setup(datalist):
    riclist1 = []
    riclist2 = []
    riclist3 = []
    for var in datalist:
        if var[2] == 1:
            riclist1.append(var)
        elif var[3] == 1:
            riclist2.append(var)
        else:
            riclist3.append(var)
    #calculating covariance of individual ric
    covariance_ric1 = calculating_covariance(riclist1)
    covariance_ric2 = calculating_covariance(riclist2)
    covariance_ric3 = calculating_covariance(riclist3)
    
    #calculating mean of individual ric
    mean_ric1 = calculating_mean(riclist1)
    mean_ric2 = calculating_mean(riclist2)
    mean_ric3 = calculating_mean(riclist3)
    # converting mean to numpy array
    initial_centroids = np.array([mean_ric1, mean_ric2, mean_ric3])
    #converting amplitude to numpy array after calculating amplitude
    amplitude = np.array([len(riclist1)/len(datalist), len(riclist2)/len(datalist), len(riclist3)/len(datalist)])
    # converting covariance to numpy array
    covariance = np.array([covariance_ric1,covariance_ric2 ,covariance_ric3])
    return initial_centroids,amplitude,covariance

# normalize function for normalising the points
def normalize(datapoint,centroids,covariance,d):
    probability = 1/pow((2*math.pi), -d/2) * pow(abs(np.linalg.det(covariance)), -1/2) * np.exp(-1/2) * np.dot(np.dot((datapoint-centroids).T, np.linalg.inv(covariance)), (datapoint-centroids))
    return probability

# highest_LLH function for calculating the maximum likelihood
def highest_LLH(data,centroids,covariance,amplitude,d,K):
    new_likelihood = 0
    for i in range(len(data)):
        temp = 0
        for k in range(K):
            temp += amplitude[k] * normalize(data[i].T,centroids[k].T, covariance[k], d)
        new_likelihood += np.log(temp)
        if DEBUG > 1: print('check temp type:',type(temp))
        
    #print("New_likelihood:",new_likelihood)
    return new_likelihood

# Expectation_step function for Exceptation algorithm 
def Expectation_step(data,centroids,covariance,amplitude,d,K,r):
    # E step
    print("Entering into Exceptation step.")
    # Calculate r[k][i], which stands for Rik
    s = np.zeros(len(data))
    for i in range(len(data)):
        temp = np.zeros(K)  # Temporary array
        # Calculate amplitude[k]*N(amplitude Xi, centroid/mean Miu(Uk), covariance/ sigma/ denoted by summation Sk) for each data[i] and the summation of that in all distributions
        for k in range(K):
            temp[k] = float(amplitude[k]) * normalize(data[i].T,centroids[k].T, covariance[k], d)
            s[i] += temp[k]
        for k in range(K):
            r[k][i] = temp[k]/s[i]
            if DEBUG > 1: print("r[k][i]=",r[k][i])

# Maximization_step function for Maximization algorithm   
def Maximization_step(data,centroids,covariance,amplitude,d,K,r):
    #M step
    print("Entering into Maximization step.")
    for k in range(K):
        # Calculate amplitude[k]
        amplitude[k] = np.sum(r[k]) / len(data)
        # Calculate centroid (denoted as mu[k])
        total = np.zeros(centroids.shape[1])
        for i in range(len(data)):
            total += r[k][i]* data[i]
        centroids[k] = total / np.sum(r[k])
        # Calculate covariance (denoted as sigma[k])
        summ = np.zeros([d,d])
        for i in range(len(data)):
            if data[i].ndim == 1:
                #used reshape to convert into matrix form
                data_temp = data[i].reshape(data.shape[1], 1)
                centriod_temp = centroids[k].reshape(centroids.shape[1], 1)
                diff_temp = data_temp - centriod_temp
                summ += r[k][i] * np.dot(diff_temp, diff_temp.T)
            else:
                summ += r[k][i] * np.dot(data[i]-centroids[i], (data[i]-centroids[i]).T)
            
        if DEBUG > 0: print("summ =",summ,"; np.sum(r[k]) =",np.sum(r[k]))
        covariance[k] = summ / np.sum(r[k])
        if DEBUG > 0: print("sigma[k]=",covariance[k])   


# gaussion_mixture function is calling all other function defined above and printing
#the result of amplitude, mean and covariance
def gaussion_mixture(data,centroids,amplitude,covariance,d,K,r):
    likelihood=None
    likelihood_threshold = 1e-3
    
    new_lld = highest_LLH(data,centroids,covariance,amplitude,d,K)
    recursion = 0
    # while condition is used for concurrently executing Estimation and Maximization
    while((recursion == 0) or ((new_lld - likelihood) > likelihood_threshold)):
        likelihood = new_lld
        Expectation_step(data,centroids,covariance,amplitude,d,K,r)
        Maximization_step(data,centroids,covariance,amplitude,d,K,r)
        new_lld = highest_LLH(data,centroids,covariance,amplitude,d,K)
        recursion += 1
    #print("Recursion time:", recursion)
    print("\nThe likelihood is:",likelihood)
    print("\nThe amplitudes are:")
    for i in amplitude:
        print(i)
    print("\nThe means are:")
    for i in centroids:
        print(i)
    print("\nThe covariances are:")
    for i in covariance:
        print(i)
 
    
#Please change parameters of data and k here if needed.
# main function where we have initialized value of k=3 
# passed the text file and have called gausion_mixture function.
def main():  
    K = 3
    data = read_data('clusters.txt')
    d = dim(data)
    datalist = initialize_data_ric(data)
    datalist = np.asarray(datalist, dtype=np.float32)
    initial_centroids,amplitude,covariance = initial_setup(datalist)
    r = np.zeros([K,len(data)])
    gaussion_mixture(data,initial_centroids,amplitude,covariance,d,K,r)

    return 0


if __name__ == '__main__':
    main()
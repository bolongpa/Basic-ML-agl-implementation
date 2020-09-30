'''Implementation of PCA in python'''

#importing modules
import numpy as np
import matplotlib.pyplot as plt

def read_data():  # to read the given data
    data = []
    with open('pca-data.txt', 'r') as f:
        for line in f:
            line = line[:-1]  # remove '\n'
            line = line.split('\t')
            # convert data to float and initialize the nearest centroid index with -1
            data.append([float(i) for i in line])
    return data

def plot_graph(data,d):
    if d == 2:
        data = data.T
        # x-axis values 
        x = data[0]
        # y-axis values 
        y = data[1] 
        
        # plotting points as a scatter plot 
        plt.scatter(x, y, label= "points", color= "blue",  
                    marker= ".", s=30) 
          
        # x-axis label 
        plt.xlabel('x - axis') 
        # frequency label 
        plt.ylabel('y - axis') 
        # plot title 
        plt.title('Data in 2 dimension') 
        # showing legend 
        plt.legend() 
        # function to show the plot 
        plt.show() 
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        data = data.T
        # x-axis values 
        x = data[0]
        # y-axis values 
        y = data[1] 
        # z-axis values 
        z = data[1] 
        
        # plotting points as a scatter plot 
        ax.scatter(x, y,z, c='y', label= "points", marker= ".") 
          
        # x-axis label 
        ax.set_xlabel('X Label')
        # y-axis label 
        ax.set_ylabel('Y Label')
        # z-axis label 
        ax.set_zlabel('Z Label') 
        # plot title 
        plt.title('Data in 3 dimension') 
        # showing legend 
        plt.legend() 
        # function to show the plot 
        plt.show()

def main():
     # define a matrix
    A = np.array(data)
    row,col = A.shape
    print("Data of n Dimension : \n",A)
    plot_graph(A,col)
    # calculate the mean of each column
    means = np.mean(A, axis = 0)
    print("Mean : \n",means)
    # calculate the Xi-Mean for every data point
    X = A - means
    print("X = Data - Mean: \n",X)
    # calculating the Covariance
    covariance = np.cov(X,rowvar=0)
    print("Covariance : \n ",covariance)
    # calculating the Eigen Values and Eigen Vectors
    Eigen_value, Eigen_vectors = np.linalg.eig(covariance)
    print("Eigen Vector : \n", Eigen_vectors)
    print("Eigen Value : \n",Eigen_value)
    #Truncating the Eigen vectors
    Eigen_vectors_truncated=np.ones((col,k))
    z=np.ones((row,k))
    for i in range(0,col):
        Eigen_vectors_truncated[i:]=Eigen_vectors[i,0:k]
    print("Eigen Vector after Truncating : \n",Eigen_vectors_truncated)
    z = np.dot(Eigen_vectors_truncated.T,A.T)
    print("PCA Models: \n",z)
    z=z.T
    print("Final Data of k Dimension : \n",z)
    #code to generate a csv file of datapoints in k dimensions where k is 2D in our case
    np.savetxt("pca-data-2D.csv", z, delimiter=",")
    plot_graph(z,k)
    return 0
    
if __name__ == '__main__':
    data = read_data()
    k = 2 #k is user defined
    main()
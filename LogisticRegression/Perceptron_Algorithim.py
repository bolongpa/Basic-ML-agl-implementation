'''Implementation of Perceptron Algorithim in python'''


#importing modules
# Load required libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
import time
start_time = time.time()



def plot(X, Y, W):
    '''
    Plot data into 3D space for showing classification.
    Plot the 3D plane that segments the datas.
    '''
    fig = plt.figure()
    
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    

    for i in range(len(X)):
        x = X[i][1]
        y = X[i][2]
        z = X[i][3]
        if Y[i] == 1:
            _c_p = ax.scatter(x, y, z, c='b', marker='.') 
        else:
            _r_p = ax.scatter(x, y, z, c='y', marker='*')

    ax.legend ([_c_p, _r_p], ['Label data = 1','Label data = -1'])
# =============================================================================
#     # create a mesh to plot in
#     x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
#     #z_min, z_max = X[:, 3].min() - 1, X[:, 3].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
# =============================================================================
    
# meshgrid used to generate a plane to classify both the labels 
    xx, yy = np.meshgrid(np.arange(-0.2, 1.2, 0.02), np.arange(-0.2, 1.2, 0.02))
    zz = -( W[0] + W[1] * xx + W[2] * yy) / W[3]
    # Plot the segmentation plane
    ax.plot_surface(xx, yy, zz, color = 'b', alpha = 0.3)
    plt.show()

clsfic_dataset = open("classification.txt", "r")
clsfic_content = clsfic_dataset.read().splitlines()

# seperated the column and have considered second last column as label 

data_list=[]
label=[]
for i in clsfic_content:
    line = [1]
    line.append(i.split(',')[0])
    line.append(i.split(',')[1])
    line.append(i.split(',')[2])
    label.append(float(i.split(',')[3]))
    data_list.append([float(x) for x in line])
    
# random function to generate 3 different initial weights  
# threshold value as 0.01    
randm = np.random.rand(3)
thrs = 0.01
wt = [thrs,randm[0],randm[1],randm[2]]
wt = np.array(wt)

# no. of iterations equal to 7000, learning rate = 0.01

max_no_iter = 7000
lr_rate = 0.01
totalviolated = 0

data=np.array(data_list) 
label = np.array(label)
for i in range(max_no_iter):
    val = []
    for x in range(len(data)):
        val.append(data[x][0]*wt[0]+data[x][1]*wt[1]+data[x][2]*wt[2]+data[x][3]*wt[3])
    
    count = 0
    for d in range(len(val)):
        #validating conditions for both the cases if values of product of data ponits and weight is 
        # greater or less and zero and if label is "+1" or "-1"
        if val[d]>=0.0 and label[d]==1:
            count+=1
        elif val[d]<0.0 and label[d]==-1:
            count+=1
        else:
            pass
    # terminating condition
    if count==len(data):
        break
    else:
        y = 0
        #calculating dot product of each data point with it's corresponding weight
        for x in data:   
            if((np.dot(x,wt)<0.0) and label[y]==1):
                for z in range(len(x)):
                    wt[z] = wt[z] + (lr_rate * x[z])
            elif((np.dot(x,wt)>=0.0) and label[y]==-1):
                for z in range(len(x)):
                    wt[z] = wt[z] - (lr_rate * x[z])
            else:
                totalviolated += 1
            y+=1

Predicted_label=np.multiply(data, wt)
Predicted_label=Predicted_label.sum(axis=1, dtype='float')
Predicted_label=np.sign(Predicted_label)

print('Final weights: ',wt[0],',',wt[1],',',wt[2],',',wt[3])

print('Number of constraints satisfied: ',count)
print('Number of iterations: ',i)

print('Accuracy: %.2f' % accuracy_score(label, Predicted_label))


#Plotting the data in 3D space
plot(data, label, wt)

print("--- Total time to execute is %s seconds ---" % (time.time() - start_time))
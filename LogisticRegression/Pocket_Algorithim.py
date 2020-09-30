'''Implementation of Pocket Algorithim in python'''

#importing modules
# Load required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
from mpl_toolkits.mplot3d import Axes3D
start_time = time.time()

#loading the dataset/text file
clsfic_dataset = open("classification.txt", "r")
clsfic_content = clsfic_dataset.read().splitlines()

data_list=[]
label=[]
# seperated the column and have considered last column as label 
for i in clsfic_content:
    line = [1]
    line.append(i.split(',')[0])
    line.append(i.split(',')[1])
    line.append(i.split(',')[2])
    label.append(float(i.split(',')[4]))
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
xlist = []
ylist = []
minimum_count = 0
minimum_wt = []
minimum_i = 0

data=np.array(data_list) 
label = np.array(label)
for i in range(max_no_iter):
    val = []
    for x in range(len(data)):
        val.append(data[x][0]*wt[0]+data[x][1]*wt[1]+data[x][2]*wt[2]+data[x][3]*wt[3])
    
    # initializing count and pending so that we can keep a count of violated cases for all 2000 tupules
    count = 0
    pending = 0
    for d in range(len(val)):
        #validating conditions for both the cases if values of product of data ponits and weight is 
        # greater or less and zero and if label is "+1" or "-1"
        if val[d]>=0.0 and label[d]==1:
            count+=1
        elif val[d]<0.0 and label[d]==-1:
            count+=1
        else:
            pending +=1
    xlist.append(i)
    ylist.append(pending)
    if count > minimum_count:
        minimum_count = count
        minimum_wt = wt
        minimum_i = i
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

Predicted_label=np.multiply(data, minimum_wt)
Predicted_label=Predicted_label.sum(axis=1, dtype='float')
Predicted_label=np.sign(Predicted_label)
            
print('Weights where we had least violated constraints : ',minimum_wt[0],',',minimum_wt[1],',',minimum_wt[2],',',minimum_wt[3])

print('Least amount of violated constraints : ', 2000-minimum_count)

print('Iteration number where least number of constraints were misclassified or violated: ',minimum_i)

print('Accuracy: %.2f' % accuracy_score(label, Predicted_label))

plt.plot(xlist,ylist)
plt.xlabel('# Iterations')
plt.ylabel('# Violated constraints')
plt.show()

print("--- Total time to execute is %s seconds ---" % (time.time() - start_time))
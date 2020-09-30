'''Implementation of SVM for linearly separable data points using QPP solver in python'''

import numpy as np
import cvxopt
import matplotlib.pyplot as plt

data=np.loadtxt('linsep.txt',dtype='float',delimiter=',')

points=data[:,0:2]  
#print(points)
labels=data[:,2]
#print(labels)
#points, labels = read_data_lin()
rows, cols = points.shape
print(points.shape)
print(cols)

# calculation of QPP using cvxopt
# parameters of QPP --> P, q, A, b, G, h, alphas
Q = np.zeros((rows, rows))
for i in range(rows):
    for j in range(rows):
        Q[i,j] = np.dot(points[i], points[j])
P = cvxopt.matrix(np.outer(labels,labels) * Q)
q = cvxopt.matrix(np.ones(rows) * -1)
A = cvxopt.matrix(labels, (1,rows))
b = cvxopt.matrix(0.0)
G = cvxopt.matrix(np.diag(np.ones(rows) * -1))
h = cvxopt.matrix(np.zeros(rows))
alphas = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)['x']).reshape(1,rows)[0]
support_vector_indices = np.where(alphas>0.00001)[0]

alphas = alphas[support_vector_indices]

support_vectors = points[support_vector_indices]
support_vectors_y = labels[support_vector_indices]

print("%d support vectors out of %d points" % (len(alphas), rows))
# calculating weights
weights = np.zeros(cols)
for i in range(len(alphas)):
    weights += alphas[i] * support_vectors_y[i] * support_vectors[i]
#calculation of intercept of a line
intercept = support_vectors_y[0] - np.dot(weights, support_vectors[0])

print('Plane formula: y = {} * x - ({})'.format((-weights[0]/weights[1]), (intercept) / weights[1]))
print("support_vectors", support_vectors)
print("Intercept:")
print(intercept)
print("Weights:")
print(weights)
plt.scatter(points[:,0],points[:,1],c=labels,cmap='bwr',alpha=1,s=50,edgecolors='k')
x2_lefttargeth = -(weights[0]*(-1)+intercept)/weights[1]
x2_righttargeth = -(weights[0]*(1)+intercept)/weights[1]
plt.scatter(support_vectors[:,0],support_vectors[:,1],facecolors='none',s=100, edgecolors='k',marker = "s")
plt.plot([-1,1], [x2_lefttargeth,x2_righttargeth])
plt.show()	

'''Implementation of SVM for non-linearly separable data points using kernel function and QPP solver in python'''

#importing modules
# Load required libraries

import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import sqrt
from mpl_toolkits.mplot3d import Axes3D 



data=np.loadtxt('nonlinsep.txt',dtype='float',delimiter=',')
points=data[:,0:2]
labels=data[:,2]
rows, cols = points.shape
#print(points)
#print(points.shape)
#print(cols)

original_data = points
# calculate new dimensions for all data points
# by squaring each point and adding it
new_dimensions = []
for point in range(len(points)):
    temporary_array = [(points[point, 0]) ** 2, (points[point, 1]) ** 2]
    new_dimensions.append(temporary_array)
new_dimensions = np.array(new_dimensions)

points= new_dimensions
#print(points)

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
######################################
original_support_vectors = original_data[support_vector_indices]
support_vectors = points[support_vector_indices]
support_vectors_y = labels[support_vector_indices]
print("%d support vectors out of %d points" % (len(alphas), rows))
print(alphas)
# calculating weights
weights = np.zeros(cols)
for i in range(len(alphas)):
    weights += alphas[i] * support_vectors_y[i] * support_vectors[i]

#calculation of intercept of a line
intercept = support_vectors_y[0] - np.dot(weights, support_vectors[0])

print("support vector indices: ", support_vector_indices)
print("original_support_vectors", original_support_vectors)
print("support_vectors", support_vectors)
print("Kernel function: polynomial, degree=2")
print("Intercept:")
print(intercept)
print("Weights:")
print(weights)
print("Equation of curve: 1.0186053938 x2 + y2 = 104.7087518504")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
# plot transformed space
ax[0].scatter(points[:,0],points[:,1],c=labels,cmap='bwr',alpha=1,s=50,edgecolors='k')
xx =range(-20,140)
yy = [(-weights[1]/weights[0])*x-intercept/weights[0] for x in xx]
ax[0].scatter(support_vectors[:,0],support_vectors[:,1],facecolors='none',s=100, edgecolors='k')
ax[0].plot(xx, yy)
# plot original space
ax[1].scatter(data[:, 0], data[:, 1], c=data[:, 2], s=30, zorder=10, cmap='RdYlBu')  # plot original points
width = 2 * sqrt(-intercept/weights[0]) * sqrt(weights[1]/weights[0])
height = 2 * sqrt(-intercept/weights[0])
curve = patches.Ellipse((0, 0), width, height, zorder=0, fc='yellow')
ax[1].add_patch(curve)
# plot support vectors
SV_origin = np.array([list(data[i, 0:2]) for i in support_vector_indices])
ax[1].scatter(SV_origin[:,0], SV_origin[:, 1], c='w', s=80, zorder=5, edgecolors='k')

plt.show()



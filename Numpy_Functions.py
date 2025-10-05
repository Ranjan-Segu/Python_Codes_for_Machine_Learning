# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 17:40:54 2025

@author: Ranjan Segu
"""

                               ####NUMPY_FUNCTIONS####
                               

#Importing_Libraries

import numpy as np

from math import pi


#Arrays

L = [1, 2, 3, 4]

A = np.array(L)

DA = np.array([[1, 2, 3],[4, 5, 6]])


A[1]
A[0:2]
DA[1,1]

A > 3
A * 2
A + np.array([5, 6, 7, 8])

DA.shape
np.append(DA, A)
np.insert(A, 1, 5)
np.delete(A, [1])
np.mean(A)
np.median(A)
np.std(A)
np.corrcoef(A)


#Initial_Placeholders

np.zeros((3,4))
np.ones((2,3,4))
d = np.arange(3,12,3)
np.linspace(0, 2, 4)    
e = np.full((3,3), 7)
f = np.eye(3)
np.random.random((4,3))
e=np.empty((3,2))


#Saving_&_Loading

np.save('my_array', A)
np.savez("array.npz", A, DA)
np.load("array.npz")

np.loadtxt("example.txt")
np.genfromtxt("diabetes.csv", delimiter=",")
np.savetxt("my_array.txt", A, delimiter=" ")
np.savetxt("my_darray.txt", DA, delimiter=" ")

#Data_Types

np.int64
np.float32
DA.dtype
np.object
np.unicode_()
np.str_()


#Inspecting_The_Array

A.shape
len(e)
DA.ndim
DA.size
DA.dtype
DA.dtype.name
A.astype(bool)

np.info(np.ndarray.dtype)


#Array_Arthimatic

d - DA
np.subtract(d, DA)
d + DA
np.add(d, DA)
d / DA
np.divide(d, DA)
d * DA
np.multiply(d, DA)

np.exp(d)
np.sqrt(d)
np.sin(d)
np.log(d)
e.dot(f)


#Comparison

d == DA
d < DA
np.array_equal(e, f)

#Aggregrate_Functions

DA.sum()
DA.min()
DA.max(axis=0)
DA.cumsum(axis=0)
d.mean()
np.median(d)
np.corrcoef(d)
np.std(d)


#Copying_&_Sorting

h = A.view()
np.copy(A)
h = A.copy()

f.sort()
f.sort(axis=0)


#Array_Manipulation

DA[1,:]
A[::-1]
A[A<2]


i = np.transpose(DA)
i.T

DA.ravel()
DA.reshape(3, -4)
h.resize((2,6))
np.append(A, h)
np.insert(A, 4, 7)
np.delete(A, [3])
np.concatenate((d, A), axis=0)
np.stack((f,e))
np.r_[e,f]
np.hstack((e,f))
np.column_stack((e,f))
np.c_[e,f]
np.hsplit(A, 2)
X=np.vsplit(DA, 2)
print(X)
Y=np.array(X)
print(Y)

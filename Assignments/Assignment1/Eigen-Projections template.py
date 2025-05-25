#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:28:58 2022

@author: 
"""
#Continum mechanics
#Eigen projections of second order tensor

import numpy as np
import cmath
from scipy.linalg import expm #for verification
import sys

A=np.array([[1,1,4],[1,3,4],[4,4,-2]]) 
B=np.array([[1,1,1],[0,2,2],[0,0,3]]) 
C=np.array([[2.,0,0],[0,2.,0],[0,0,2.]]) 


T=B #T is the matrix to be evaluated



# The principal invariants


# calc eigen values
EigenV=np.array([0.,0.,0.])
for i in range(3):
    if (J1**2-3*J2==0): #Special case
                
    else:


# 2. perturbed eigenvalues, in the case of multiple occurring eigenvalues
# => compare EV 1<->2, 1<->3, and 2<->3 in the following request
tol=1e-6
delta=1e-6


# Print the computed eigenvalues


#
#Calc D_i

#Calc P_i


#For evaluation
# sum(P_i) should be the identity matrix


#sum(lambda_i * P_i) has to be = T: Should be the original matrix T

#Calculate exp(T)








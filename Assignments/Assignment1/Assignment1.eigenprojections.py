# %% [markdown]
# # Assignment 1: Eigen Projections
# Ali Jahangiri

# %% [markdown]
# In this assignment, we wish to calculate the eigen projections of a second order tensor. To do so, first we will import the required libraries.

# %%
!pip install numpy
!pip install scipy
import numpy as np
import cmath
from scipy.linalg import expm #for verification
import sys

# %% [markdown]
# ## Part A: Eigenvalue Computations

# %% [markdown]
# We define three Numpy arrays as follows:
# 
# $$
# \begin{cases}
# A=
# \begin{pmatrix}
# 1 & 1 &  4 \\
# 1 & 3 &  4 \\
# 4 & 4 & -2 
# \end{pmatrix} \\
# B=
# \begin{pmatrix}
# 1 & 1 &  1 \\
# 0 & 2 &  2 \\
# 0 & 0 &  3 
# \end{pmatrix} \\
# C=
# \begin{pmatrix}
# 2 & 0 &  0 \\
# 0 & 2 &  0 \\
# 0 & 0 &  2 
# \end{pmatrix}
# \end{cases}
# $$

# %%
A=np.array([[1,1,4],[1,3,4],[4,4,-2]]) 
B=np.array([[1,1,1],[0,2,2],[0,0,3]]) 
C=np.array([[2.,0,0],[0,2.,0],[0,0,2.]]) 

# %% [markdown]
# With the matrix $A, B$ and $C$ determined, we now specify the variable $T$ to point to matrix $B$. Later, we will create a function, but for now the notebook will walk through the steps for Tensor B.

# %%
T=B

# %% [markdown]
# We now, progress to calculate the values for $J_1$, $J_2$ and $J_3$
# 
# $$
# \begin{cases}
# J_1 = \mathrm{tr}(T)\\
# J_2 = \frac{1}{2}[J_1^2 - \mathrm{T^2}] \\
# J_3 = \det T
# \end{cases}
# $$

# %%
I = np.eye(3)
J1 = np.trace(T)
J2 = 0.5 * (J1**2 - np.trace(np.dot(T, T)))
J3 = np.linalg.det(T)

# %% [markdown]
# `EigenV` is defined to store the values for the three eigen values. In general, this is determined by the formula
# 
# $$
# \lambda_k = \frac{1}{3}
# \left[
#     J_1 + 2\sqrt(J_1^2-3J_2) \cos{\frac{1}{3}(\phi + 2\pi [k-1])}
# \right]
# $$
# where
# $$
# \phi = \arccos{\frac{2J_1^3-9J_1J_2+27J_3}{2(J_1^2-3J_2)^{3/2}}}
# $$
# 
# However, in the case where $J_1^2-3J_2 = 0$, $\lambda_k$ is not obtainable via the above formula. Instead we have
# 
# $$
# \lambda_k = \frac{1}{3}J_1 +
# \frac{1}{3}\left[ 
#     27J_3-J_1^3
# \right]^\frac{1}{3}
# \left[
#     \cos{\frac{2}{3} \pi k}
# \right]
# $$

# %%
EigenV=np.array([0.,0.,0.])

if J1**2-3*J2 == 0:
    for k in range(1,4):
        EigenV[k-1] = (1/3) * J1 + (1/3) * (27*J3 - J1**3)**(1/3) * np.cos((2/3)*np.pi*k)
else:
    phi = np.arccos((2*J1**3 - 9*J1*J2 + 27*J3) / (2 * np.sqrt((J1**2 - 3*J2)**3)))
    for k in range(1,4):
        EigenV[k-1] = (1/3) * (J1 + 2 * np.sqrt(J1**2 - 3*J2) * np.cos((phi + 2*np.pi*k)/3))  #corrected multiplication      

# %% [markdown]
# In another special case where we have duplicate eigenvalues, we pertubate them. We define $\lambda_1$ and $\lambda_2$ as being duplicate when this condition holds 
# 
# $$
# \frac{|\lambda_i - \lambda_j|}{\max{|\lambda_1|, |\lambda_2|,|\lambda_3|}}
# < \delta
# $$
# 
# In case that this inequality holds (based on tolerance set in code below) we have the following eigenvalues
# 
# $$
# \begin{cases}
# \lambda_i = \lambda_1 (1+\delta) \\
# \lambda_j = \lambda_1 (1-\delta) \\
# \lambda_k = \frac{\lambda_k}{(1+\delta)(1-\delta)}
# \end{cases}
# $$

# %%
#specify conditions
delta = 1e-6 # don't need tol, delta does it too :)
maxEig = np.max(np.abs(EigenV)) #denominator

if maxEig == 0:
    print("Eigenvalues are zero, fix the code.")
    sys.exit()

if abs(EigenV[0]-EigenV[1])/maxEig < delta:
    EigenV[0] = EigenV[0] * (1 + delta)
    EigenV[1] = EigenV[1] * (1 - delta)
    EigenV[2] = EigenV[2] / ((1 + delta)*(1-delta))
elif abs(EigenV[0]-EigenV[2])/maxEig < delta:
    EigenV[0] = EigenV[0] * (1 + delta)
    EigenV[1] = EigenV[1] / ((1 + delta)*(1-delta))
    EigenV[2] = EigenV[2] * (1 - delta)
elif abs(EigenV[1]-EigenV[2])/maxEig < delta:
    EigenV[0] = EigenV[0] / ((1 + delta)*(1-delta))
    EigenV[1] = EigenV[1] * (1 + delta)
    EigenV[2] = EigenV[2] * (1 - delta)

#print the magic
print("Eigenvalues:", EigenV)


# %% [markdown]
# ## Part B: Calculate Eigenprojection

# %% [markdown]
# We now have the eigenvalue matrix values! 🎉
# To calculate the eigenprojections, first we must calculate the products of the eigenvalue differences. 
# 
# $$
# D_i = \prod_{j=\frac{1}{3}}^3 [\lambda_i - \lambda_j]
# $$

# %%
D = np.array([0.,0.,0.])
#can we use np.prod?
D[0] = (EigenV[0] - EigenV[1]) * (EigenV[0] - EigenV[2])
D[1] = (EigenV[1] - EigenV[0]) * (EigenV[1] - EigenV[2])
D[2] = (EigenV[2] - EigenV[0]) * (EigenV[2] - EigenV[1])

print("Differences:", D)

# %% [markdown]
# Phew! Nearly there. Next and final step is to put together the pieces of the eigen projection. 
# 
# $$
# P_i = \frac{1}{D_i}\prod_{j=\frac{1}{i}}^3 [T-\lambda_j I]
# $$

# %%
P = [] #empty array
for i in range(3):
    temp = I.copy() # identity matrix
    for j in range(3):
        if i != j: 
            temp = np.dot(temp, (T - EigenV[j] * I))
    P.append(temp / D[i])

#print("Eigenprojections", P) # how to make it pretty

from pprint import pprint
print("P_1=")
pprint(P[0])
print("P_2=")
pprint(P[0]) 
print("P_3=")
pprint(P[0]) 

# %% [markdown]
# ## Part C: Verification

# %% [markdown]
# To make this testable, we will define a function, and throw everthing we did so far into it. 

# %%
def calcEP(T):
    I = np.eye(3)
    J1 = np.trace(T)
    J2 = 0.5 * (J1**2 - np.trace(np.dot(T, T)))
    J3 = np.linalg.det(T)
    EigenV = np.array([0.,0.,0.])
    if J1**2-3*J2 == 0:
        for k in range(1,4):
            EigenV[k-1] = (1/3) * J1 + (1/3) * (27*J3 - J1**3)**(1/3) * np.cos((2/3)*np.pi*k)
    else:
        phi = np.arccos((2*J1**3 - 9*J1*J2 + 27*J3) / (2 * np.sqrt((J1**2 - 3*J2)**3)))
        for k in range(1,4):
            EigenV[k-1] = (1/3) * (J1 + 2 * np.sqrt(J1**2 - 3*J2) * np.cos((phi + 2*np.pi*k)/3))  #corrected multiplication
    #specify conditions
    delta = 1e-6 # don't need tol, delta does it too :)
    maxEig = np.max(np.abs(EigenV)) #denominator

    if maxEig == 0:
        print("Eigenvalues are zero, fix the code.")
        sys.exit()

    if abs(EigenV[0]-EigenV[1])/maxEig < delta:
        EigenV[0] = EigenV[0] * (1 + delta)
        EigenV[1] = EigenV[1] * (1 - delta)
        EigenV[2] = EigenV[2] / ((1 + delta)*(1-delta))
    elif abs(EigenV[0]-EigenV[2])/maxEig < delta:
        EigenV[0] = EigenV[0] * (1 + delta)
        EigenV[1] = EigenV[1] / ((1 + delta)*(1-delta))
        EigenV[2] = EigenV[2] * (1 - delta)
    elif abs(EigenV[1]-EigenV[2])/maxEig < delta:
        EigenV[0] = EigenV[0] / ((1 + delta)*(1-delta))
        EigenV[1] = EigenV[1] * (1 + delta)
        EigenV[2] = EigenV[2] * (1 - delta)
    
    D = np.array([0.,0.,0.])
    #can we use np.prod?
    D[0] = (EigenV[0] - EigenV[1]) * (EigenV[0] - EigenV[2])
    D[1] = (EigenV[1] - EigenV[0]) * (EigenV[1] - EigenV[2])
    D[2] = (EigenV[2] - EigenV[0]) * (EigenV[2] - EigenV[1])

    P = [] #empty array
    for i in range(3):
        temp = I.copy() # identity matrix
        for j in range(3):
            if i != j: 
                temp = np.dot(temp, (T - EigenV[j] * I))
        P.append(temp / D[i])
        
    return EigenV, P

# %% [markdown]
# We can uncomment the line below, to select the tensor.

# %%
#T=A
T=B
#T=C

EigenV, P = calcEP(T)

# %% [markdown]
# We can check to see if the conditions below are met
# 
# $$
# \begin{cases}
# I=\sum_{i=1}^3 P_i \\
# T=\sum_{i=1}^3\lambda_i P_i
# \end{cases}
# $$
# 
# The `numpy.testing` module has a wonderful option for checking to see if two values are nearly equal called `assert_almost_equal`, this is great since this (a) automates the process for checking and (b) accounts for floating point errors. 😃

# %%
# sum(P_i) should be the identity matrix
sum_P = np.sum(np.array(P), axis=0)
print("Sum of projections:")
pprint(sum_P)

print("Are sum_P and I NOT (almost) equal?")
print(np.testing.assert_almost_equal(sum_P, I, err_msg="Sum of projections is not equal to identity matrix!"))

# %%
# sum(lambda_i * P_i) has to be = T: Should be the original matrix T
sum_lambda_P = np.sum(np.array([EigenV[i] * P[i] for i in range(3)]), axis=0)
print("Sum of (eigenvalue * projection):")
pprint(sum_lambda_P)
print("Original matrix T:", T)

print("Are sum_lambda_P and T NOT (almost) equal?")
print(np.testing.assert_almost_equal(sum_P, I, err_msg="sum(lambda_i * P_i) has to be = T: Should be the original matrix T!"))

# %% [markdown]
# Finally, as an additional verification step we calculate $e^T$ where
# 
# $$
# e^T=\sum e^{\lambda_i} P_i
# $$
# 
# We verify this with the `scipy.linalg.expm` method.

# %%
# Calculate exp(T) 
exp_T_spectral = np.sum(np.array([np.exp(EigenV[i]) * P[i] for i in range(3)]), axis=0)
exp_T_scipy = expm(T)

print("exp(T) using spectral decomp:")
pprint(exp_T_spectral)
print("exp(T) using scipy:")
pprint(exp_T_scipy)


print("Are exp_T_spectral and exp_T_scipy NOT (almost) equal?")
print(np.testing.assert_almost_equal(exp_T_spectral, exp_T_scipy, err_msg=""))



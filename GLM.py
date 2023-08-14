#!/usr/bin/env python
# coding: utf-8

# In[53]:


#Lasso回归1
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


# Define the psi function and its derivative
def psi(x):
    return x**2

# Define the loss functions for the Transferring and Debiasing steps
def loss_function(w, X, y, lambda_w):
    linear_term = -y.T @ X @ w
    nonlinear_term = np.sum([psi(w.T @ x_i) for x_i in X])
    penalty_term = lambda_w * np.sum(np.abs(w))
    return (linear_term + nonlinear_term + penalty_term) / (X.shape[0] + X.shape[1])

def trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set):
    X0, y0 = target_data
    Xa, ya = np.concatenate([source_data[i][0] for i in transferring_set], axis=0), np.concatenate([source_data[i][1] for i in transferring_set], axis=0)

    # Transferring step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for w
        args=(np.concatenate([X0, Xa]), np.concatenate([y0, ya]), lambda_w),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    w_hat = result.x

    # Debiasing step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for delta
        args=(X0, y0, lambda_delta),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    delta_hat = result.x

    beta_hat = w_hat + delta_hat

    return beta_hat

#创建稀疏矩阵
def sprase_rand(m, n, p):
    if type(m) != int or type(n) != int:
        raise TypeError('Rows(m) and Columns(n) must be an interger!')
    if p <= 0 or p > 1:
        raise ValueError('p must in (0, 1] !')
    # Counts of non-zero elements in sparse-matrix
    count = int(m * n * p)
    # Indexs of non-zero elements in sparse-matrix
    rows = np.random.randint(0, m, count)
    cols = np.random.randint(0, n, count)
    # Values of non-zero elements in sparse-matrix
    # (Gaussian distribution)
    data = np.random.randn(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(m, n))

p = 1


# Generate synthetic data for testing
np.random.seed(0)
n_features = 5
n_samples_target = 100
n_samples_source = 200
n_source_datasets_max = 20  # maximum number of source datasets
n_runs = 20  # number of runs for each case
n_P = 10

 
X0_sprase = sprase_rand(n_samples_target,n_features, p)
X0 = X0_sprase.toarray()
w_true = np.random.normal(0, 1, n_features)
y0 = X0 @ w_true + np.random.normal(0, 0.1, n_samples_target)

target_data = (X0, y0) 


lambda_w = 1
lambda_delta = 0.1
p1 = 0 
errors = []
for n_source_datasets in range(1, n_P):
    p1 = p1+0.1
    errors_case = []
    for _ in range(n_runs): 
        source_data = []
        W = []  # List to store the weights of each source
        for _ in range(1, n_source_datasets_max + 1):
            X_source_sprase =sprase_rand(n_samples_source,n_features, p1)
            X_source = X_source_sprase.toarray()
            W_source = np.random.normal(0, 1, n_features)
            y_source = X_source @ W_source + np.random.normal(0, 0.1, n_samples_source)
            source_data.append((X_source, y_source))
            W.append(W_source)  # Here we assume that the true weight is used as the weight of each source

        # Define the h value
        #h = 20  # You can change this value as needed
        # Compute the contrast for each source
        contrasts = [np.linalg.norm(w_true - w_k, ord=1) for w_k in W]
        h = np.mean(contrasts)
        # Define the level-h transferring set
        transferring_set = [k for k, contrast in enumerate(contrasts) if contrast <= h] 
        # Run the algorithm and calculate the error
        beta_hat = trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set)
        error = np.linalg.norm(beta_hat - w_true)
        errors_case.append(error)
    errors.append(errors_case)

# Print the errors for each case
errors_per_case = []
for i, errors_case in enumerate(errors):
    errors_per_case.append((i+1, np.mean(errors_case)))

errors_per_case
print(errors_per_case)
print(errors)


# Extract data for plotting
source_datasets_numbers = [result[0] for result in errors_per_case]
mean_errors = [result[1] for result in errors_per_case]
error_stddevs = [np.std(errors_case) for errors_case in errors]

# Create the plot
plt.figure(figsize=(10, 6))
#plt.errorbar(source_datasets_numbers, mean_errors,  fmt='o', color='orange', elinewidth=3, capsize=0)
plt.plot(source_datasets_numbers, mean_errors, '-o', color='blue',label = 'Trans_Lasso')  # Added line
plt.legend()
plt.xlabel('Number of source datasets')
plt.ylabel('L2 error')
plt.title('Linear Regression for different number of source datasets（h=20）')
plt.show()  


# In[30]:


print(contrasts)
print(transferring_set)
c = np.mean(contrasts)
print(c)


# In[52]:


print(mean_errors)


# In[45]:


#Poisson regression2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.sparse import coo_matrix


# Define the psi function and its derivative for Poisson regression
def psi_poisson(x):
    return np.exp(x)

# Define the loss function for the Transferring and Debiasing steps
def loss_function(w, X, y, lambda_w, psi):
    linear_term = -y.T @ X @ w
    nonlinear_term = np.sum([psi(w.T @ x_i) for x_i in X])
    penalty_term = lambda_w * np.sum(np.abs(w))
    return (linear_term + nonlinear_term + penalty_term) / (X.shape[0] + X.shape[1])

def trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set, psi):
    X0, y0 = target_data
    Xa, ya = np.concatenate([source_data[i][0] for i in transferring_set], axis=0), np.concatenate([source_data[i][1] for i in transferring_set], axis=0)

    # Transferring step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for w
        args=(np.concatenate([X0, Xa]), np.concatenate([y0, ya]), lambda_w, psi),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    w_hat = result.x

    # Debiasing step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for delta
        args=(X0, y0, lambda_delta, psi),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    delta_hat = result.x

    beta_hat = w_hat + delta_hat

    return beta_hat

#创建稀疏矩阵
def sprase_rand(m, n, p):
    if type(m) != int or type(n) != int:
        raise TypeError('Rows(m) and Columns(n) must be an interger!')
    if p <= 0 or p > 1:
        raise ValueError('p must in (0, 1] !')
    # Counts of non-zero elements in sparse-matrix
    count = int(m * n * p)
    # Indexs of non-zero elements in sparse-matrix
    rows = np.random.randint(0, m, count)
    cols = np.random.randint(0, n, count)
    # Values of non-zero elements in sparse-matrix
    # (Gaussian distribution)
    data = np.random.randn(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(m, n))

p = 1


# Generate synthetic data for testing
np.random.seed(0)
n_features = 5
n_samples_target = 100
n_samples_source = 200
n_source_datasets_max = 20  # maximum number of source datasets
n_runs = 20  # number of runs for each case
n_P = 10

 
X0_sprase = sprase_rand(n_samples_target,n_features, p)
X0 = X0_sprase.toarray()
w_true = np.random.normal(0, 1, n_features)
y0 = X0 @ w_true + np.random.normal(0, 0.1, n_samples_target)
target_data = (X0, y0) 


lambda_w = 1
lambda_delta = 0.1
p1 = 0 
errors = []
for n_source_datasets in range(1, n_P):
    p1 = p1+0.1
    errors_case = []
    for _ in range(n_runs): 
        source_data = []
        W = []  # List to store the weights of each source
        for _ in range(1, n_source_datasets_max + 1):
            X_source_sprase =sprase_rand(n_samples_source,n_features, p1)
            X_source = X_source_sprase.toarray()
            W_source = np.random.normal(0, 1, n_features)
            y_source = X_source @ W_source + np.random.normal(0, 0.1, n_samples_source)
            source_data.append((X_source, y_source))
            W.append(W_source)  # Here we assume that the true weight is used as the weight of each source

        # Define the h value
        #h = 20  # You can change this value as needed
        # Compute the contrast for each source
        contrasts = [np.linalg.norm(w_true - w_k, ord=1) for w_k in W]
        h = np.mean(contrasts)
        # Define the level-h transferring set
        transferring_set = [k for k, contrast in enumerate(contrasts) if contrast <= h] 
        # Run the algorithm and calculate the error
        beta_hat = trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set, psi_poisson)
        error = np.linalg.norm(beta_hat - w_true)
        errors_case.append(error)
    errors.append(errors_case)

# Print the errors for each case
errors_per_case = []
for i, errors_case in enumerate(errors):
    errors_per_case.append((i+1, np.mean(errors_case)))

errors_per_case
print(errors_per_case)
#print(errors)


# Extract data for plotting
source_datasets_numbers_p = [result[0] for result in errors_per_case]
mean_errors_p = [result[1] for result in errors_per_case]

# Create the plot
plt.figure(figsize=(10, 6))
#plt.errorbar(source_datasets_numbers, mean_errors,  fmt='o', color='orange', elinewidth=3, capsize=0)
plt.plot(source_datasets_numbers_p, mean_errors_p, '-^', color='red',label = 'Trans_Possion')  # Added line
plt.legend()
plt.xlabel('Number of source datasets')
plt.ylabel('L2 error')
plt.title('Linear Regression for different number of source datasets（h=20）')
plt.show()  


# In[48]:


#Logistic regression3
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.sparse import coo_matrix


# Define the psi function and its derivative for Logistic regression
def psi_logistic(x):
    return expit(x)

# Define the loss function for the Transferring and Debiasing steps
def loss_function(w, X, y, lambda_w, psi):
    linear_term = -y.T @ X @ w
    nonlinear_term = np.sum([psi(w.T @ x_i) for x_i in X])
    penalty_term = lambda_w * np.sum(np.abs(w))
    return (linear_term + nonlinear_term ) / (X.shape[0] + X.shape[1])

def trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set, psi):
    X0, y0 = target_data
    Xa, ya = np.concatenate([source_data[i][0] for i in transferring_set], axis=0), np.concatenate([source_data[i][1] for i in transferring_set], axis=0)

    # Transferring step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for w
        args=(np.concatenate([X0, Xa]), np.concatenate([y0, ya]), lambda_w, psi),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    w_hat = result.x

    # Debiasing step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for delta
        args=(X0, y0, lambda_delta, psi),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    delta_hat = result.x

    beta_hat = w_hat + delta_hat

    return beta_hat

#创建稀疏矩阵
def sprase_rand(m, n, p):
    if type(m) != int or type(n) != int:
        raise TypeError('Rows(m) and Columns(n) must be an interger!')
    if p <= 0 or p > 1:
        raise ValueError('p must in (0, 1] !')
    # Counts of non-zero elements in sparse-matrix
    count = int(m * n * p)
    # Indexs of non-zero elements in sparse-matrix
    rows = np.random.randint(0, m, count)
    cols = np.random.randint(0, n, count)
    # Values of non-zero elements in sparse-matrix
    # (Gaussian distribution)
    data = np.random.randn(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(m, n))

p = 1


# Generate synthetic data for testing
np.random.seed(0)
n_features = 5
n_samples_target = 100
n_samples_source = 200
n_source_datasets_max = 10  # maximum number of source datasets
n_runs = 20  # number of runs for each case
n_P = 10

 
X0_sprase = sprase_rand(n_samples_target,n_features, p)
X0 = X0_sprase.toarray()
w_true = np.random.normal(0, 1, n_features)
y0_continuous = X0 @ w_true + np.random.normal(0, 0.1, n_samples_target)
y0 = np.where(y0_continuous > np.median(y0_continuous), 1, 0)

target_data = (X0, y0) 


lambda_w = 1
lambda_delta = 0.1
p1 = 0 
errors = []
for n_source_datasets in range(1, n_P):
    p1 = p1+0.1
    errors_case = []
    for _ in range(n_runs): 
        source_data = []
        W = []  # List to store the weights of each source
        for _ in range(1, n_source_datasets_max + 1):
            X_source_sprase =sprase_rand(n_samples_source,n_features, p1)
            X_source = X_source_sprase.toarray()
            W_source = np.random.normal(0, 1, n_features)
            y_source_continuous = X_source @ W_source + np.random.normal(0, 0.1, n_samples_source)
            y_source = np.where(y_source_continuous > np.median(y_source_continuous), 1, 0)
            source_data.append((X_source, y_source))
            W.append(W_source)  # Here we assume that the true weight is used as the weight of each source

        # Define the h value
        #h = 20  # You can change this value as needed
        # Compute the contrast for each source
        contrasts = [np.linalg.norm(w_true - w_k, ord=1) for w_k in W]
        h = np.mean(contrasts)
        # Define the level-h transferring set
        transferring_set = [k for k, contrast in enumerate(contrasts) if contrast <= h] 
        # Run the algorithm and calculate the error
        beta_hat = trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set, psi_logistic)
        error = np.linalg.norm(beta_hat - w_true)
        errors_case.append(error)
    errors.append(errors_case)

# Print the errors for each case
errors_per_case = []
for i, errors_case in enumerate(errors):
    errors_per_case.append((i+1, np.mean(errors_case)))

errors_per_case
print(errors_per_case)
#print(errors)


# Extract data for plotting
source_datasets_numbers_L = [result[0] for result in errors_per_case]
mean_errors_L = [result[1] for result in errors_per_case]

# Create the plot
plt.figure(figsize=(10, 6))
#plt.errorbar(source_datasets_numbers, mean_errors,  fmt='o', color='orange', elinewidth=3, capsize=0)
plt.plot(source_datasets_numbers_L, mean_errors_L, '-^', color='red',label = 'Trans_Logit')  # Added line
plt.legend()
plt.xlabel('Number of source datasets')
plt.ylabel('L2 error')
plt.title('Linear Regression for different number of source datasets（h=20）')
plt.show()  


# In[47]:


print(mean_errors_L)


# In[56]:


#Ridge回归4
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


# Define the psi function and its derivative
def psi(x):
    return x**2

# Define the loss functions for the Transferring and Debiasing steps
def loss_function(w, X, y, lambda_w):
    linear_term = -y.T @ X @ w
    nonlinear_term = np.sum([psi(w.T @ x_i) for x_i in X])
    penalty_term = lambda_w * np.mean(np.square(w))  
    return (linear_term + nonlinear_term + penalty_term) / (X.shape[0] + X.shape[1])

def trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set):
    X0, y0 = target_data
    Xa, ya = np.concatenate([source_data[i][0] for i in transferring_set], axis=0), np.concatenate([source_data[i][1] for i in transferring_set], axis=0)

    # Transferring step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for w
        args=(np.concatenate([X0, Xa]), np.concatenate([y0, ya]), lambda_w),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    w_hat = result.x

    # Debiasing step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for delta
        args=(X0, y0, lambda_delta),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    delta_hat = result.x

    beta_hat = w_hat + delta_hat

    return beta_hat

#创建稀疏矩阵
def sprase_rand(m, n, p):
    if type(m) != int or type(n) != int:
        raise TypeError('Rows(m) and Columns(n) must be an interger!')
    if p <= 0 or p > 1:
        raise ValueError('p must in (0, 1] !')
    # Counts of non-zero elements in sparse-matrix
    count = int(m * n * p)
    # Indexs of non-zero elements in sparse-matrix
    rows = np.random.randint(0, m, count)
    cols = np.random.randint(0, n, count)
    # Values of non-zero elements in sparse-matrix
    # (Gaussian distribution)
    data = np.random.randn(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(m, n))

p = 1


# Generate synthetic data for testing
np.random.seed(99)
n_features = 5
n_samples_target = 100
n_samples_source = 200
n_source_datasets_max = 20  # maximum number of source datasets
n_runs = 20  # number of runs for each case
n_P = 10

 
X0_sprase = sprase_rand(n_samples_target,n_features, p)
X0 = X0_sprase.toarray()
w_true = np.random.normal(0, 1, n_features)
y0 = X0 @ w_true + np.random.normal(0, 0.1, n_samples_target)

target_data = (X0, y0) 


lambda_w = 1
lambda_delta = 1
p1 = 0 
errors = []
for n_source_datasets in range(1, n_P):
    p1 = p1+0.1
    errors_case = []
    for _ in range(n_runs): 
        source_data = []
        W = []  # List to store the weights of each source
        for _ in range(1, n_source_datasets_max + 1):
            X_source_sprase =sprase_rand(n_samples_source,n_features, p1)
            X_source = X_source_sprase.toarray()
            W_source = np.random.normal(0, 1, n_features)
            y_source = X_source @ W_source + np.random.normal(0, 0.1, n_samples_source)
            source_data.append((X_source, y_source))
            W.append(W_source)  # Here we assume that the true weight is used as the weight of each source

        # Define the h value
        #h = 20  # You can change this value as needed
        # Compute the contrast for each source
        contrasts = [np.linalg.norm(w_true - w_k, ord=1) for w_k in W]
        h = np.mean(contrasts)
        # Define the level-h transferring set
        transferring_set = [k for k, contrast in enumerate(contrasts) if contrast <= h] 
        # Run the algorithm and calculate the error
        beta_hat = trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set)
        error = np.linalg.norm(beta_hat - w_true)
        errors_case.append(error)
    errors.append(errors_case)

# Print the errors for each case
errors_per_case = []
for i, errors_case in enumerate(errors):
    errors_per_case.append((i+1, np.mean(errors_case)))

errors_per_case
print(errors_per_case)

# Extract data for plotting
source_datasets_numbers_r = [result[0] for result in errors_per_case]
mean_errors_r = [result[1] for result in errors_per_case]

# Create the plot
plt.figure(figsize=(10, 6))
#plt.errorbar(source_datasets_numbers, mean_errors,  fmt='o', color='orange', elinewidth=3, capsize=0)
plt.plot(source_datasets_numbers_r, mean_errors_r, '-o', color='g',label = 'Trans_ridge')  # Added line
plt.legend()
plt.xlabel('Number of source datasets')
plt.ylabel('L2 error')
plt.title('Linear Regression for different number of source datasets（h=20）')
plt.show()  


# In[60]:


#Elastic Net Regression5
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


# Define the psi function and its derivative
def psi(x):
    return x**2

# Define the loss functions for the Transferring and Debiasing steps
def loss_function(w, X, y, lambda_w):
    linear_term = -y.T @ X @ w
    nonlinear_term = np.sum([psi(w.T @ x_i) for x_i in X])
    penalty_term =lambda_w * np.sum(np.abs(w)) + 0.6 * np.mean(np.square(w))  
    return (linear_term + nonlinear_term + penalty_term) / (X.shape[0] + X.shape[1])

def trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set):
    X0, y0 = target_data
    Xa, ya = np.concatenate([source_data[i][0] for i in transferring_set], axis=0), np.concatenate([source_data[i][1] for i in transferring_set], axis=0)

    # Transferring step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for w
        args=(np.concatenate([X0, Xa]), np.concatenate([y0, ya]), lambda_w),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    w_hat = result.x

    # Debiasing step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for delta
        args=(X0, y0, lambda_delta),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    delta_hat = result.x

    beta_hat = w_hat + delta_hat

    return beta_hat

#创建稀疏矩阵
def sprase_rand(m, n, p):
    if type(m) != int or type(n) != int:
        raise TypeError('Rows(m) and Columns(n) must be an interger!')
    if p <= 0 or p > 1:
        raise ValueError('p must in (0, 1] !')
    # Counts of non-zero elements in sparse-matrix
    count = int(m * n * p)
    # Indexs of non-zero elements in sparse-matrix
    rows = np.random.randint(0, m, count)
    cols = np.random.randint(0, n, count)
    # Values of non-zero elements in sparse-matrix
    # (Gaussian distribution)
    data = np.random.randn(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(m, n))

p = 1


# Generate synthetic data for testing
np.random.seed(9)
n_features = 5
n_samples_target = 100
n_samples_source = 200
n_source_datasets_max = 20  # maximum number of source datasets
n_runs = 20  # number of runs for each case
n_P = 10

 
X0_sprase = sprase_rand(n_samples_target,n_features, p)
X0 = X0_sprase.toarray()
w_true = np.random.normal(0, 1, n_features)
y0 = X0 @ w_true + np.random.normal(0, 0.1, n_samples_target)

target_data = (X0, y0) 


lambda_w = 0.3
lambda_delta =0.3
p1 = 0 
errors = []
for n_source_datasets in range(1, n_P):
    p1 = p1+0.1
    errors_case = []
    for _ in range(n_runs): 
        source_data = []
        W = []  # List to store the weights of each source
        for _ in range(1, n_source_datasets_max + 1):
            X_source_sprase =sprase_rand(n_samples_source,n_features, p1)
            X_source = X_source_sprase.toarray()
            W_source = np.random.normal(0, 1, n_features)
            y_source = X_source @ W_source + np.random.normal(0, 0.1, n_samples_source)
            source_data.append((X_source, y_source))
            W.append(W_source)  # Here we assume that the true weight is used as the weight of each source

        # Define the h value
        #h = 20  # You can change this value as needed
        # Compute the contrast for each source
        contrasts = [np.linalg.norm(w_true - w_k, ord=1) for w_k in W]
        h = np.mean(contrasts)
        # Define the level-h transferring set
        transferring_set = [k for k, contrast in enumerate(contrasts) if contrast <= h] 
        # Run the algorithm and calculate the error
        beta_hat = trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set)
        error = np.linalg.norm(beta_hat - w_true)
        errors_case.append(error)
    errors.append(errors_case)

# Print the errors for each case
errors_per_case = []
for i, errors_case in enumerate(errors):
    errors_per_case.append((i+1, np.mean(errors_case)))

errors_per_case
print(errors_per_case)

# Extract data for plotting
source_datasets_numbers_EN =  [result[0] for result in errors_per_case]
mean_errors_EN = [result[1] for result in errors_per_case]
# Create the plot
plt.figure(figsize=(10, 6))
#plt.errorbar(source_datasets_numbers, mean_errors,  fmt='o', color='orange', elinewidth=3, capsize=0)
plt.plot(source_datasets_numbers_EN, mean_errors_EN, '-o', color='c',label = 'Trans_EN')  # Added line
plt.legend()
plt.xlabel('Number of source datasets')
plt.ylabel('L2 error')
plt.title('Linear Regression for different number of source datasets（h=20）')
plt.show()  


# In[66]:


#SCAD Regression6
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


# Define the psi function and its derivative
def psi(x):
    return x**2 

lambda1 = 2 
# Define the loss functions for the Transferring and Debiasing steps
def loss_function(w, X, y, lambda_w):
    linear_term = -y.T @ X @ w
    nonlinear_term = np.sum([psi(w.T @ x_i) for x_i in X])
    penalty_term = np.sum(np.sign(w)*(np.abs(w)-lambda1))
    return (linear_term + nonlinear_term + penalty_term) / (X.shape[0] + X.shape[1])

def trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set):
    X0, y0 = target_data
    Xa, ya = np.concatenate([source_data[i][0] for i in transferring_set], axis=0), np.concatenate([source_data[i][1] for i in transferring_set], axis=0)

    # Transferring step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for w
        args=(np.concatenate([X0, Xa]), np.concatenate([y0, ya]), lambda_w),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    w_hat = result.x

    # Debiasing step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for delta
        args=(X0, y0, lambda_delta),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    delta_hat = result.x

    beta_hat = w_hat + delta_hat

    return beta_hat

#创建稀疏矩阵
def sprase_rand(m, n, p):
    if type(m) != int or type(n) != int:
        raise TypeError('Rows(m) and Columns(n) must be an interger!')
    if p <= 0 or p > 1:
        raise ValueError('p must in (0, 1] !')
    # Counts of non-zero elements in sparse-matrix
    count = int(m * n * p)
    # Indexs of non-zero elements in sparse-matrix
    rows = np.random.randint(0, m, count)
    cols = np.random.randint(0, n, count)
    # Values of non-zero elements in sparse-matrix
    # (Gaussian distribution)
    data = np.random.randn(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(m, n))

p = 1


# Generate synthetic data for testing
np.random.seed(3)
n_features = 5
n_samples_target = 100
n_samples_source = 200
n_source_datasets_max = 20  # maximum number of source datasets
n_runs = 20  # number of runs for each case
n_P = 10

 
X0_sprase = sprase_rand(n_samples_target,n_features, p)
X0 = X0_sprase.toarray() 
w_true = np.random.normal(0, 1, n_features)
y0 = X0 @ w_true + np.random.normal(0, 0.1, n_samples_target)

target_data = (X0, y0) 


lambda_w = 0.3
lambda_delta =0.3
p1 = 0 
errors = []
for n_source_datasets in range(1, n_P):
    p1 = p1+0.1
    errors_case = []
    for _ in range(n_runs): 
        source_data = []
        W = []  # List to store the weights of each source
        for _ in range(1, n_source_datasets_max + 1):
            X_source_sprase =sprase_rand(n_samples_source,n_features, p1)
            X_source = X_source_sprase.toarray()
            W_source = np.random.normal(0, 1, n_features)
            y_source = X_source @ W_source + np.random.normal(0, 0.1, n_samples_source)
            source_data.append((X_source, y_source))
            W.append(W_source)  # Here we assume that the true weight is used as the weight of each source

        # Define the h value
        #h = 20  # You can change this value as needed
        # Compute the contrast for each source
        contrasts = [np.linalg.norm(w_true - w_k, ord=1) for w_k in W]
        h = np.mean(contrasts)
        # Define the level-h transferring set
        transferring_set = [k for k, contrast in enumerate(contrasts) if contrast <= h] 
        # Run the algorithm and calculate the error
        beta_hat = trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set)
        error = np.linalg.norm(beta_hat - w_true)
        errors_case.append(error)
    errors.append(errors_case)

# Print the errors for each case
errors_per_case = []
for i, errors_case in enumerate(errors):
    errors_per_case.append((i+1, np.mean(errors_case)))

errors_per_case
print(errors_per_case)

# Extract data for plotting
source_datasets_numbers_SCAD =  [result[0] for result in errors_per_case]
mean_errors_SCAD = [result[1] for result in errors_per_case] 
# Create the plot
plt.figure(figsize=(10, 6))
#plt.errorbar(source_datasets_numbers, mean_errors,  fmt='o', color='orange', elinewidth=3, capsize=0)
plt.plot(source_datasets_numbers_SCAD, mean_errors_SCAD, '-o', color='c',label = 'Trans_SCAD')  # Added line
plt.legend()
plt.xlabel('Number of source datasets')
plt.ylabel('L2 error')
plt.title('Linear Regression for different number of source datasets（h=20）')
plt.show()  


# In[69]:


#dantzig selector7
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


# Define the psi function and its derivative
def psi(x):
    return x**2

# Define the loss functions for the Transferring and Debiasing steps
def loss_function(w, X, y, lambda_w):
    linear_term = -y.T @ X @ w
    nonlinear_term = np.sum([psi(w.T @ x_i) for x_i in X])
    penalty_term =np.linalg.norm(w)  
    return (linear_term + nonlinear_term + penalty_term) / (X.shape[0] + X.shape[1])

def trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set):
    X0, y0 = target_data
    Xa, ya = np.concatenate([source_data[i][0] for i in transferring_set], axis=0), np.concatenate([source_data[i][1] for i in transferring_set], axis=0)

    # Transferring step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for w
        args=(np.concatenate([X0, Xa]), np.concatenate([y0, ya]), lambda_w),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    w_hat = result.x

    # Debiasing step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for delta
        args=(X0, y0, lambda_delta),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    delta_hat = result.x

    beta_hat = w_hat + delta_hat

    return beta_hat

#创建稀疏矩阵
def sprase_rand(m, n, p):
    if type(m) != int or type(n) != int:
        raise TypeError('Rows(m) and Columns(n) must be an interger!')
    if p <= 0 or p > 1:
        raise ValueError('p must in (0, 1] !')
    # Counts of non-zero elements in sparse-matrix
    count = int(m * n * p)
    # Indexs of non-zero elements in sparse-matrix
    rows = np.random.randint(0, m, count)
    cols = np.random.randint(0, n, count)
    # Values of non-zero elements in sparse-matrix
    # (Gaussian distribution)
    data = np.random.randn(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(m, n))

p = 1


# Generate synthetic data for testing
np.random.seed(20) 
n_features = 5
n_samples_target = 100
n_samples_source = 200
n_source_datasets_max = 20  # maximum number of source datasets
n_runs = 20  # number of runs for each case
n_P = 10

 
X0_sprase = sprase_rand(n_samples_target,n_features, p)
X0 = X0_sprase.toarray()
w_true = np.random.normal(0, 1, n_features)
y0 = X0 @ w_true + np.random.normal(0, 0.1, n_samples_target)

target_data = (X0, y0) 


lambda_w = 0.3
lambda_delta =0.3
p1 = 0 
errors = []
for n_source_datasets in range(1, n_P):
    p1 = p1+0.1
    errors_case = []
    for _ in range(n_runs): 
        source_data = []
        W = []  # List to store the weights of each source
        for _ in range(1, n_source_datasets_max + 1):
            X_source_sprase =sprase_rand(n_samples_source,n_features, p1)
            X_source = X_source_sprase.toarray()
            W_source = np.random.normal(0, 1, n_features)
            y_source = X_source @ W_source + np.random.normal(0, 0.1, n_samples_source)
            source_data.append((X_source, y_source))
            W.append(W_source)  # Here we assume that the true weight is used as the weight of each source

        # Define the h value
        #h = 20  # You can change this value as needed
        # Compute the contrast for each source
        contrasts = [np.linalg.norm(w_true - w_k, ord=1) for w_k in W]
        h = np.mean(contrasts)
        # Define the level-h transferring set
        transferring_set = [k for k, contrast in enumerate(contrasts) if contrast <= h] 
        # Run the algorithm and calculate the error
        beta_hat = trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set)
        error = np.linalg.norm(beta_hat - w_true)
        errors_case.append(error)
    errors.append(errors_case)

# Print the errors for each case
errors_per_case = []
for i, errors_case in enumerate(errors):
    errors_per_case.append((i+1, np.mean(errors_case)))

errors_per_case
print(errors_per_case)

# Extract data for plotting
source_datasets_numbers_D =  [result[0] for result in errors_per_case]
mean_errors_D = [result[1] for result in errors_per_case] 
# Create the plot
plt.figure(figsize=(10, 6))
#plt.errorbar(source_datasets_numbers, mean_errors,  fmt='o', color='orange', elinewidth=3, capsize=0)
plt.plot(source_datasets_numbers_D, mean_errors_D, '-o', color='c',label = 'Trans_dantzig')  # Added line
plt.legend()
plt.xlabel('Number of source datasets')
plt.ylabel('L2 error')
plt.title('Linear Regression for different number of source datasets（h=20）')
plt.show()  


# In[81]:


#Adaptive LASSO 8
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


# Define the psi function and its derivative
def psi(x):
    return x**2

# Define the loss functions for the Transferring and Debiasing steps
def loss_function(w, X, y, lambda_w):
    linear_term = -y.T @ X @ w
    nonlinear_term = np.sum([psi(w.T @ x_i) for x_i in X])
    penalty_term = lambda_w * np.sum(np.abs(w))
    return (linear_term + nonlinear_term + penalty_term) / (X.shape[0] + X.shape[1])

def trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set):
    X0, y0 = target_data
    Xa, ya = np.concatenate([source_data[i][0] for i in transferring_set], axis=0), np.concatenate([source_data[i][1] for i in transferring_set], axis=0)

    # Transferring step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for w
        args=(np.concatenate([X0, Xa]), np.concatenate([y0, ya]), lambda_w),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    w_hat = result.x

    # Debiasing step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for delta
        args=(X0, y0, lambda_delta),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    delta_hat = result.x

    beta_hat = w_hat + delta_hat

    return beta_hat

#创建稀疏矩阵
def sprase_rand(m, n, p):
    if type(m) != int or type(n) != int:
        raise TypeError('Rows(m) and Columns(n) must be an interger!')
    if p <= 0 or p > 1:
        raise ValueError('p must in (0, 1] !')
    # Counts of non-zero elements in sparse-matrix
    count = int(m * n * p)
    # Indexs of non-zero elements in sparse-matrix
    rows = np.random.randint(0, m, count)
    cols = np.random.randint(0, n, count)
    # Values of non-zero elements in sparse-matrix
    # (Gaussian distribution)
    data = np.random.randn(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(m, n))

p = 1


# Generate synthetic data for testing
#np.random.seed(1)
n_features = 5
n_samples_target = 100
n_samples_source = 200
n_source_datasets_max = 20  # maximum number of source datasets
n_runs = 20  # number of runs for each case
n_P = 10

 
X0_sprase = sprase_rand(n_samples_target,n_features, p)
X0 = X0_sprase.toarray()
w_true = np.random.normal(0, 1, n_features)
y0 = X0 @ w_true + np.random.normal(0, 0.1, n_samples_target)

target_data = (X0, y0) 
error1 = 0 

p1 = 0 
errors = []
for n_source_datasets in range(1, n_P):
    p1 = p1+0.1
    errors_case = []
    for _ in range(n_runs): 
        if error1==0 :
            lambda_w = 0.1
            lambda_delta = 0.1
        elif error_past<error1: 
            lambda_w = lambda_w+0.1
            lambda_delta =lambda_delta+0.1
        
        source_data = []
        W = []  # List to store the weights of each source
        for _ in range(1, n_source_datasets_max + 1):
            X_source_sprase =sprase_rand(n_samples_source,n_features, p1)
            X_source = X_source_sprase.toarray()
            W_source = np.random.normal(0, 1, n_features)
            y_source = X_source @ W_source + np.random.normal(0, 0.1, n_samples_source)
            source_data.append((X_source, y_source))
            W.append(W_source)  # Here we assume that the true weight is used as the weight of each source

        # Define the h value
        #h = 20  # You can change this value as needed
        # Compute the contrast for each source
        contrasts = [np.linalg.norm(w_true - w_k, ord=1) for w_k in W]
        h = np.mean(contrasts)
        # Define the level-h transferring set
        transferring_set = [k for k, contrast in enumerate(contrasts) if contrast <= h] 
        # Run the algorithm and calculate the error
        beta_hat = trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set)
        error1 = np.linalg.norm(beta_hat - w_true)
        errors_case.append(error1)
        error_past = np.mean(errors_case)
    errors.append(errors_case)

# Print the errors for each case
errors_per_case = []
for i, errors_case in enumerate(errors):
    errors_per_case.append((i+1, np.mean(errors_case)))

errors_per_case
print(errors_per_case)



# Extract data for plotting
source_datasets_numbers_adlasso = [result[0] for result in errors_per_case]
mean_errors_adlasso = [result[1] for result in errors_per_case]


# Create the plot
plt.figure(figsize=(10, 6))
#plt.errorbar(source_datasets_numbers, mean_errors,  fmt='o', color='orange', elinewidth=3, capsize=0)
plt.plot(source_datasets_numbers_adlasso, mean_errors_adlasso, '-o', color='blue',label = 'Trans_ad_Lasso')  # Added line
plt.legend()
plt.xlabel('Number of source datasets')
plt.ylabel('L2 error')
plt.title('Linear Regression for different number of source datasets（h=20）')
plt.show()  


# In[83]:





# In[91]:


# AD Ridge回归9岭
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


# Define the psi function and its derivative
def psi(x):
    return x**2

# Define the loss functions for the Transferring and Debiasing steps
def loss_function(w, X, y, lambda_w):
    linear_term = -y.T @ X @ w
    nonlinear_term = np.sum([psi(w.T @ x_i) for x_i in X])
    penalty_term = lambda_w * np.mean(np.square(w))  
    return (linear_term + nonlinear_term + penalty_term) / (X.shape[0] + X.shape[1])

def trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set):
    X0, y0 = target_data
    Xa, ya = np.concatenate([source_data[i][0] for i in transferring_set], axis=0), np.concatenate([source_data[i][1] for i in transferring_set], axis=0)

    # Transferring step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for w
        args=(np.concatenate([X0, Xa]), np.concatenate([y0, ya]), lambda_w),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    w_hat = result.x

    # Debiasing step
    result = minimize(
        fun=loss_function,
        x0=np.zeros(X0.shape[1]),  # initial guess for delta
        args=(X0, y0, lambda_delta),
        method='L-BFGS-B'  # this method can handle the L1 penalty
    )
    delta_hat = result.x

    beta_hat = w_hat + delta_hat

    return beta_hat

#创建稀疏矩阵
def sprase_rand(m, n, p):
    if type(m) != int or type(n) != int:
        raise TypeError('Rows(m) and Columns(n) must be an interger!')
    if p <= 0 or p > 1:
        raise ValueError('p must in (0, 1] !')
    # Counts of non-zero elements in sparse-matrix
    count = int(m * n * p)
    # Indexs of non-zero elements in sparse-matrix
    rows = np.random.randint(0, m, count)
    cols = np.random.randint(0, n, count)
    # Values of non-zero elements in sparse-matrix
    # (Gaussian distribution)
    data = np.random.randn(len(rows))
    return coo_matrix((data, (rows, cols)), shape=(m, n))

p = 0.1 #取值0到1


# Generate synthetic data for testing
#np.random.seed(99)
n_features = 5
n_samples_target = 100
n_samples_source = 200
n_source_datasets_max = 20  # maximum number of source datasets
n_runs = 20  # number of runs for each case
n_P = 10

 
X0_sprase = sprase_rand(n_samples_target,n_features, p)
X0 = X0_sprase.toarray()
w_true = np.random.normal(0, 1, n_features)
y0 = X0 @ w_true + np.random.normal(0, 0.1, n_samples_target)

target_data = (X0, y0) 

error2 = 0 
p1 = 0 
errors = []
for n_source_datasets in range(1, n_P):
    p1 = p1+0.1
    errors_case = []
    for _ in range(n_runs): 
        if error2==0 :
            lambda_w = 0.1
            lambda_delta = 0.1
        elif error_past1<error2: 
            lambda_w = lambda_w+0.01
            lambda_delta =lambda_delta+0.01
            
        source_data = []
        W = []  # List to store the weights of each source
        for _ in range(1, n_source_datasets_max + 1):
            X_source_sprase =sprase_rand(n_samples_source,n_features, p1)
            X_source = X_source_sprase.toarray()
            W_source = np.random.normal(0, 1, n_features)
            y_source = X_source @ W_source + np.random.normal(0, 0.1, n_samples_source)
            source_data.append((X_source, y_source))
            W.append(W_source)  # Here we assume that the true weight is used as the weight of each source

        # Define the h value
        #h = 20  # You can change this value as needed
        # Compute the contrast for each source
        contrasts = [np.linalg.norm(w_true - w_k, ord=1) for w_k in W]
        h = np.mean(contrasts)
        # Define the level-h transferring set
        transferring_set = [k for k, contrast in enumerate(contrasts) if contrast <= h] 
        # Run the algorithm and calculate the error
        beta_hat = trans_glm(target_data, source_data, lambda_w, lambda_delta, transferring_set)
        error2 = np.linalg.norm(beta_hat - w_true)
        errors_case.append(error2)
        error_past1 = np.mean(errors_case)
    errors.append(errors_case)

# Print the errors for each case
errors_per_case = []
for i, errors_case in enumerate(errors):
    errors_per_case.append((i+1, np.mean(errors_case)))

errors_per_case
print(errors_per_case)

# Extract data for plotting
source_datasets_numbers_ADr = [result[0] for result in errors_per_case]
mean_errors_ADr = [result[1] for result in errors_per_case]

# Create the plot
plt.figure(figsize=(10, 6))
#plt.errorbar(source_datasets_numbers, mean_errors,  fmt='o', color='orange', elinewidth=3, capsize=0)
plt.plot(source_datasets_numbers_ADr, mean_errors_ADr, '-o', color='g',label = 'Trans_AD_ridge')  # Added line
plt.legend()
plt.xlabel('Number of source datasets')
plt.ylabel('L2 error')
plt.title('Linear Regression for different number of source datasets（h=20）')
plt.show()  


# In[103]:


X0_sprase = sprase_rand(10,5, 0.1)
X0 = X0_sprase.toarray()
print(X0)


# In[101]:


plt.figure(figsize=(10, 6))
#plt.errorbar(source_datasets_numbers, mean_errors,  fmt='o', color='orange', elinewidth=3, capsize=0)
plt.plot(source_datasets_numbers, mean_errors, '-o', color='blue',label = 'Trans_Lasso')  # Added line
plt.plot(source_datasets_numbers_p, mean_errors_p, '-^', color='red',label = 'Trans_Possion')  # Added line
plt.plot(source_datasets_numbers_r, mean_errors_r, '-x', color='g',label = 'Trans_ridge')  # Added line
plt.plot(source_datasets_numbers_EN, mean_errors_EN, '->', color='c',label = 'Trans_EN')  # Added line
plt.legend()
plt.xlabel('Number of source datasets')
plt.ylabel('L2 error')
plt.title('Linear Regression for different number of source datasets（h=20）')
plt.show()  


# In[100]:


plt.figure(figsize=(10, 6))
#plt.errorbar(source_datasets_numbers, mean_errors,  fmt='o', color='orange', elinewidth=3, capsize=0)
plt.plot(source_datasets_numbers_SCAD, mean_errors_SCAD, '-o', color='r',label = 'Trans_SCAD')  # Added line
plt.plot(source_datasets_numbers_D, mean_errors_D, '-^', color='b',label = 'Trans_dantzig')  # Added line
plt.plot(source_datasets_numbers_adlasso, mean_errors_adlasso, '-x', color='g',label = 'Trans_ad_Lasso')  # Added line
plt.plot(source_datasets_numbers_ADr, mean_errors_ADr, '-v', color='c',label = 'Trans_AD_ridge')  # Added line
plt.legend()
plt.xlabel('Number of source datasets')
plt.ylabel('L2 error')
plt.title('Linear Regression for different number of source datasets（h=mean）')
plt.show()  


# In[ ]:





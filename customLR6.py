import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import pandas as pd

data = np.genfromtxt("Data//home.txt",delimiter=",")

# print(data.shape) ##(47, 3)

# print(X.shape) #(47, 2)
# print(y.shape) #(47, 1)

data = normalize(data,axis=0)
# print(data.shape) #(47, 3)
X = data[:,0:2]
y = data[:,2:]

#Hyperparameters

##Batch Gradient Descent
learning_rate = 0.09
max_itr = 500

##Stochastic Gradient Descent
s_learning_rate = 0.005 # Reduced slightly for better stability
s_max_itr = 500

##mini-batch Gradient Descent
mb_learning_rate = 0.09
mb_max_itr = 500
batch_size = 16

tempX = np.ones((X.shape[0], X.shape[1]+1))
# print(tempX.shape) #(47, 3)
tempX[:,1:] = X
theta = np.zeros((X.shape[1]+1,1))
s_theta = np.zeros((X.shape[1]+1,1)) ## (3,1)
mb_theta = np.zeros((X.shape[1]+1,1)) ## (3,1)
# print(theta)
# print(theta.shape) ## (3,1)
# print(tempX.shape) ## (47,3)

##Hypothesis
def h(theta,X):
    tempX = np.ones((X.shape[0], X.shape[1]+1))
    tempX[:,1:] = X
    # print(tempX[:3])
    # print(np.matmul(tempX,theta).shape)
    return np.matmul(tempX,theta) 

def gradient(theta,X,y):
    tempX = np.ones((X.shape[0],X.shape[1]+1))
    tempX[:,1:] = X 
    d_theta = np.average((h(theta,X)-y)*tempX,axis=0)
    # print(d_theta)
    ##[-3.40412660e+05 -7.64209128e+08 -1.12036770e+06]
    ##
    # print(d_theta.shape) ##(3,)
    d_theta = d_theta.reshape(d_theta.shape[0],1)
    # print(d_theta.shape) ##(3,1)
    return d_theta

def loss(theta,X,y):
    y_hat = h(theta,X)
    return np.average(np.square(y-y_hat))

##Batch Gradient Descent
print("\nBatch Gradient Descent \n")
def gradient_descent(theta,X,y,learning_rate,max_iteration,gap):
    cost = np.zeros(max_iteration)

    for i in range(max_iteration):
        d_theta = gradient(theta,X,y)
        theta = theta -  learning_rate*d_theta
        # print(theta)
        cost[i] = loss(theta,X,y)
        if i % gap == 0 :
            print("Iteration :", i, " loss : ",loss(theta,X,y))
    return theta,cost

theta,cost = gradient_descent(theta,X,y,learning_rate,max_itr,100)
print("final theta [Batch Gradient] : ",theta.flatten())
# print("Final cost :: ",cost)

#############Stochastic GD ################
print("\nStochastic Gradient Descent \n")
def stochastic_gradient_descent(s_theta,X,y,s_learning_rate,max_itr,gap):
    cost = np.zeros(max_itr)
    for i in range(max_itr): ##max_itr  = 500
        for j in range(X.shape[0]): ##X.shape[0] = 47
            # reshaped logic to pick 1 record at a time
            d_theta = gradient(s_theta,X[j,:].reshape(1,X.shape[1]),
                                       y[j,:].reshape(1,y.shape[1]))
            # print(X[j,:].reshape(1,X.shape[1]).shape) ##(1, 2)
            # print(y[j,:].reshape(1,y.shape[1]).shape) ##(1, 1)
            s_theta = s_theta -  s_learning_rate*d_theta
                # print(theta)
        cost[i] = loss(s_theta,X,y)
        if i % gap == 0 :
            print("Iteration :", i, " loss : ",loss(s_theta,X,y))
    return s_theta,cost

s_theta, s_cost = stochastic_gradient_descent(s_theta,X,y,s_learning_rate,s_max_itr,100)
print("final theta [Stochastic Gradient] : ",s_theta.flatten())
# print("Final cost :: ",s_cost)

###Mini-batch Gradient Descent
print("\nMini Batch Gradient Descent \n")

def minibatch_gradient_descent(theta,X,y,learning_rate,max_itr,gap):
    cost = np.zeros(max_itr)
    for i in range(max_itr):
        for j in range(0,X.shape[0],batch_size):
           d_theta =gradient(theta,X[j:j+batch_size,:],y[j:j+batch_size,:])
           theta=theta-learning_rate*d_theta

        cost[i]=loss(theta,X,y)
        if i% gap==0:
            print("Iteration : ",i,'loss :',loss(theta,X,y))
    return theta,cost

mb_theta,mb_cost = minibatch_gradient_descent(mb_theta,X,y,mb_learning_rate,mb_max_itr,100)
print("final theta [minibatch gradient] : ",mb_theta.flatten())



fig,ax = plt.subplots()
ax.plot(np.arange(max_itr),cost,'r')
ax.plot(np.arange(s_max_itr),s_cost,'b')
ax.plot(np.arange(mb_max_itr),mb_cost,'g')

ax.legend(loc='upper right', 
          labels=['Batch gradient descent',
                  'Stochastic gradient descent',
                  'mini batch gradient descent'])
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
# import numpy as np
# from sklearn.preprocessing import normalize
# import matplotlib.pyplot as plt
# import pandas as pd


# data = np.genfromtxt("Data//home.txt",delimiter=",")

# # print(data.shape) ##(47, 3)


# # print(X.shape) #(47, 2)
# # print(y.shape) #(47, 1)

# data = normalize(data,axis=0)
# # print(data.shape) #(47, 3)
# X = data[:,0:2]
# y = data[:,2:]

# #Hyperparameters

# ##Batch Gradient Descent
# learning_rate = 0.09
# max_itr = 500

# ##Stochastic Gradient Descent
# s_learning_rate = 0.06
# s_max_itr = 500

# ##mini-batch Gradient Descent
# mb_learning_rate = 0.09
# mb_max_itr = 500
# batch_size = 16


# tempX = np.ones((X.shape[0], X.shape[1]+1))
# # print(tempX.shape) #(47, 3)
# tempX[:,1:] = X
# theta = np.zeros((X.shape[1]+1,1))
# s_theta = np.zeros((X.shape[1]+1,1)) ## (3,1)
# mb_theta = np.zeros((X.shape[1]+1,1)) ## (3,1)
# # print(theta)
# # print(theta.shape) ## (3,1)
# # print(tempX.shape) ## (47,3)

# ##Hypothesis
# def h(theta,X):
#     tempX = np.ones((X.shape[0], X.shape[1]+1))
#     tempX[:,1:] = X
#     # print(tempX[:3])
#     # print(np.matmul(tempX,theta).shape)
#     return np.matmul(tempX,theta) 

# def gradient(theta,X,y):
#     tempX = np.ones((X.shape[0],X.shape[1]+1))
#     tempX[:,1:] = X 
#     d_theta = np.average((h(theta,X)-y)*tempX,axis=0)
#     # print(d_theta)
#     ##[-3.40412660e+05 -7.64209128e+08 -1.12036770e+06]
#     ##
#     # print(d_theta.shape) ##(3,)
#     d_theta = d_theta.reshape(d_theta.shape[0],1)
#     # print(d_theta.shape) ##(3,1)
#     return d_theta

# def loss(theta,X,y):
#     y_hat = h(theta,X)
#     return np.average(np.square(y-y_hat))


# ##Batch Gradient Descent
# print("\nBatch Gradient Descent \n")
# def gradient_descent(theta,X,y,learning_rate,max_iteration,gap):
#     cost = np.zeros(max_iteration)

#     for i in range(max_iteration):
#         d_theta = gradient(theta,X,y)
#         theta = theta -  learning_rate*d_theta
#         # print(theta)
#         cost[i] = loss(theta,X,y)
#         if i % gap == 0 :
#             print("Iteration :", i, " loss : ",loss(theta,X,y))
#     return theta,cost

# theta,cost = gradient_descent(theta,X,y,learning_rate,max_itr,100)
# print("final theta [Batch Gradient] : ",theta)
# # print("Final cost :: ",cost)

# #############Stochastic GD ################
# print("\nStochastic Gradient Descent \n")
# def stochastic_gradient_descent(s_theta,X,y,s_leaning_rate,max_itr,gap):
#     cost = np.zeros(max_itr)
#     for i in range(max_itr): ##max_itr  = 500
#         for j in range(X.shape[0]): ##X.shape[0] = 47
#             d_theta = gradient(s_theta,X[j,:].
#                             reshape(1,X.shape[1]),
#                             y[j,:].reshape(1,y.shape[1]))
#             # print(X[j,:].reshape(1,X.shape[1]).shape) ##(1, 2)
#             # print(y[j,:].reshape(1,y.shape[1]).shape) ##(1, 1)
#             s_theta = s_theta -  s_learning_rate*d_theta
#                 # print(theta)
#         cost[i] = loss(s_theta,X,y)
#         if i % gap == 0 :
#             print("Iteration :", i, " loss : ",loss(s_theta,X,y))
#     return s_theta,cost



# s_theta, s_cost = stochastic_gradient_descent(s_theta,X,y,s_learning_rate,s_max_itr,100)
# print("final theta [Stochastic Gradient] : ",s_theta)
# # print("Final cost :: ",s_cost)

# ###Mini-batch Gradient Descent
# print("\nMini Batch Gradient Descent \n")

# def minibatch_gradient_descent(theta,X,y,learning_rate,max_itr,gap):
#     cost = np.zeros(max_itr)
#     for i in range(max_itr):
#         for j in range(0,X.shape[0],batch_size):
#            d_theta =gradient(theta,X[j:j+batch_size,:],y[j:j+batch_size,:])
#            theta=theta-learning_rate*d_theta

#         cost[i]=loss(theta,X,y)
#         if i% gap==0:
#             print("Iteration : ",i,'loss :',loss(theta,X,y))
#     return theta,cost

# mb_theta,mb_cost = minibatch_gradient_descent(mb_theta,X,y,mb_learning_rate,mb_max_itr,100)
# print("final theta [minibatch gradient] : ",mb_theta)

# fig,ax = plt.subplots()
# ax.plot(np.arange(max_itr),cost,'r')
# ax.plot(np.arange(s_max_itr),s_cost,'b')

# ax.plot(np.arange(mb_max_itr),mb_cost,'g')

# ax.legend(loc='upper right', 
#           labels=['Batch gradient descent',
#                   'Stochastic gradient descent',
#                   'mini batch gradient descent'])
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Cost')
# ax.set_title('Error vs. Training Epoch')
# plt.show()

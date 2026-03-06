import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import normalize

data=np.genfromtxt("data/home.txt",delimiter=',')
# print(data.shape) #-> 47*3

# print(X[:5])
# print(y[:5])
# print(X.shape) #(47,2)
# print(y.shape) #(47,1)

data=normalize(data,axis=0) #scale it from 0 to 1
# print(data[:5])
# print(data.shape) #(47,3)
X=data[:,0:2]
y=data[:,2:]

#hyperparameters of batch gradient descent
learning_rate= 0.09
max_itr=500

#hyperparameters of stochastic gradient descent
s_learning_rate= 0.005 # Using a smaller rate for SGD stability
s_max_itr=500          # Changed from 1 to 500 to see convergence

tempX=np.ones((X.shape[0],X.shape[1]+1))
tempX[:, 1:] = X   # column for x1
theta= np.zeros((X.shape[1]+1,1)) #1 is shape 2 but we want 3 so +1
s_theta=np.zeros((X.shape[1]+1,1))
# print(theta)
# print(theta.shape) #(3,1)
# print("shape=",tempX.shape) #(47,3)

##hypothesis
def h(theta,X):
    tempX=np.ones((X.shape[0],X.shape[1]+1))
    tempX[:,1:] = X
    return np.matmul(tempX,theta)

def gradient(theta,X,y):   
    tempX=np.ones((X.shape[0],X.shape[1]+1))
    tempX[:,1:]=X
    d_theta=np.average((h(theta,X)-y)*tempX,axis=0)
    d_theta=d_theta.reshape(d_theta.shape[0],1)
    return d_theta

def loss(theta,X,y):
    y_hat=h(theta,X)
    return np.average(np.square(y-y_hat))

#batch gradient descent
def gradient_descent(theta,X,y,learning_rate,max_itreration,gap): #all data taken at once bcz of gd
    cost=np.zeros(max_itreration)
    for i in range(max_itreration):
        d_theta=gradient(theta,X,y)
        theta=theta-learning_rate*d_theta
        cost[i]=loss(theta,X,y)
        if i % gap==0:
            print("Iteration :",i,"loss :",loss(theta,X,y))
    return theta,cost

#stochastic GD
def stochastic_gradient_descent(theta,X,y,learning_rate,max_itr,gap):
    cost=np.zeros(max_itr)
    for i in range(max_itr):
        for j in range(X.shape[0]): #single data training
            # CHANGE: called gradient() directly and fixed y.reshape
            d_theta=gradient(theta, X[j,:].reshape(1,X.shape[1]), y[j,:].reshape(1,1)) 
            # CHANGE: update theta inside the j loop
            theta=theta-learning_rate*d_theta
            
        cost[i]=loss(theta,X,y)
        if i % gap==0:
            print("Iteration :",i,"loss :",loss(theta,X,y))
    return theta,cost

print("--- Running Batch GD ---")
theta, cost = gradient_descent(theta, X, y, learning_rate, max_itr, 100)

print("\n--- Running Stochastic GD ---")
s_theta,s_cost=stochastic_gradient_descent(s_theta,X,y,s_learning_rate,s_max_itr,100)

print("\nfinal batch theta :", theta.flatten())
print("final stochastic theta :", s_theta.flatten())



# Plotting the results
fig,ax=plt.subplots()
ax.plot(np.arange(max_itr),cost,'r', label='Batch GD')
ax.plot(np.arange(s_max_itr),s_cost,'b', label='Stochastic GD')
ax.legend(loc='upper right')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Epoch')
plt.show()


# import numpy as np
# import pandas as pd
# from sklearn import linear_model
# import matplotlib.pyplot as plt
# from sklearn import preprocessing
# from sklearn.preprocessing import normalize
# import matplotlib.pyplot as plt

# data=np.genfromtxt("data/home.txt",delimiter=',')
# # print(data.shape) #-> 47*3

# # print(X[:5])
# # print(y[:5])
# # print(X.shape) #(47,2)
# # print(y.shape) #(47,1)

# # x1=X[:,0]
# # x2=X[:,1]

# # print(x1.shape) #put data of 1st col as name x1
# # print(x2.shape)

# # min_max_scaler=preprocessing.MinMaxScaler()
# # scaled_X=min_max_scaler.fit_transform(X)
# # print(scaled_X)

# data=normalize(data,axis=0) #scale it from 0 to 1
# # print(data[:5])
# # print(data.shape) #(47,3)
# X=data[:,0:2]
# y=data[:,2:]

# #hyperparameters of batch gradient descent
# learning_rate= 0.09
# max_itr=500

# #hyperparameters of stochastic gradient descent
# s_learning_rate= 0.06
# s_max_itr=1

# # print(data.shape) #(47,3)
# # print(data.shape[0]) #47
# # print(data.shape[1]) #3 #all col are theta

# # print(np.zeros(data.shape[1])) #[0. 0. 0.]
# # temp= np.zeros(data.shape[1])

# # print(np.ones(data.shape[1])) #[1. 1. 1.]
# # tempX= np.ones(data.shape[1])
# # print(tempX.shape) #(3,)

# tempX=np.ones((X.shape[0],X.shape[1]+1))
# tempX[:, 1:] = X   # column for x1
# theta= np.zeros((X.shape[1]+1,1)) #1 is shape 2 but we want 3 so +1
# s_theta=np.zeros((X.shape[1]+1,1))
# print(theta)
# print(theta.shape) #(3,1)
# print("shape=",tempX.shape) #(47,3)
# #::::Multiplication possible:::::#

# print(np.matmul(tempX,theta).shape) ##(47,1)



# #print(tempX[:3])

# # a=np.replace[tempX,x1]
# # print(a) 

# ##hypothesis
# def h(theta,X):
#     tempX=np.ones((X.shape[0],X.shape[1]+1))
#     tempX[:,1:] = X

#     return np.matmul(tempX,theta)

# def gradient(theta,X,y):   
#     tempX=np.ones((X.shape[0],X.shape[1]+1))
#     tempX[:,1:]=X
#     # print("shape of tempX=",tempX.shape)

#     d_theta=np.average((h(theta,X)-y)*tempX,axis=0)
#     # print(d_theta)
#     # print(d_theta.shape)# (3,)
#     d_theta=d_theta.reshape(d_theta.shape[0],1)
#     # print(d_theta.shape) #(3,1)
#     return d_theta

# def loss(theta,X,y):
#     y_hat=h(theta,X)
#     return np.average(np.square(y-y_hat))

# #batch gradient descent
# def gradient_descent(theta,X,y,learning_rate,max_itreration,gap): #all data taken at once bcz of gd
#     cost=np.zeros(max_itreration)
#     for i in range(max_itreration):
#         d_theta=gradient(theta,X,y)
#         theta=theta-learning_rate*d_theta
#         # print(theta)
#         cost[i]=loss(theta,X,y)
#         if i % gap==0:
#             print("Iteration :",i,"loss :",loss(theta,X,y))
#     return theta,cost

# # theta,cost=gradient_descent(theta,X,y,learning_rate,max_itr,100)
# # print("final theta : ",theta)
# # print("final cost :",cost)

# #stochastic GD
# def stochastic_gradient_descent(theta,X,y,learning_rate,max_itr,gap):
#     cost=np.zeros(max_itr)
#     for i in range(max_itr):
#         for j in range(X.shape[0]): #single data training (47 records taken each time and loop for 500 times)
#             d_theta=gradient_descent(theta,X[j,:].reshape(1,X.shape[1]),y[j,:].reshape) #0th row and all col bcz data (47,2)
#             # print(X[j,:])
#         theta=theta-learning_rate*d_theta
#         # print(theta)
#         cost[i]=loss(theta,X,y)
#         if i % gap==0:
#             print("Iteration :",i,"loss :",loss(theta,X,y))
#     return theta,cost



# s_theta,s_cost=stochastic_gradient_descent(s_theta,X,y,s_learning_rate,s_max_itr,100)
# print("final theta :",s_theta)
# print("final cost :",s_cost)


# # fig,ax=plt.subplots()
# # ax.plot(np.arange(max_itr),cost,'r')
# # ax.legend(loc='upper right',labels=['batch gradient descent'])
# # ax.set_xlabel('Iterations')
# # ax.set_ylabel('Cost')
# # ax.set_title('Error vs Training Epoch')
# # plt.show()



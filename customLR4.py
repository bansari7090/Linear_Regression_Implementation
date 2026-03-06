
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

data=np.genfromtxt("data/data.csv",delimiter=',')
X=data[:,[0]] ##features-->convert it into 2d
# print(X)
# print(X.shape) #(100,1) : it means 2d list

y=data[:,1]
# print(y.shape) #(100,) : it means 1d list

#pandas-->but it takes 1st row as heading
# first_col=data.ilac[:,0] #all rows 1st col
# print(first_col)
# print(first_col.shape)



#Hyperparamters
learning_rate=0.0001
max_itr=100000

#training model

#hypothesis
def h(X,m,b):
    # print("b=",b)
    # print("m=",m)
    return m*X+b
def gradient(X,y,m,b):
    y_hat=h(X,m,b) #--> y_hat means y^
    # print(y_hat)
    dm=np.average((y_hat-y)*X) #derived val of m
    db=np.average((y_hat-y))
    # print("dm=",dm)
    # print("db= ",db) #initially it was 0 now it comes -ve now try to get +ve
    return dm,db
    # return 0.,0.
def loss(X,y,m,b):
    y_hat=h(X,m,b)
    return np.average(np.square(y-y_hat))
def gradient_descent(X,y,learning_rate,max_itr):
    m=0.
    b=0.
    losses=[]
    for i in range(max_itr):
        dm,db=gradient(X,y,m,b)
        # m-=learning_rate*dm
        b-=learning_rate*db
        # print("m=",m)
        # print("b=",b)
        loss_value=loss(X,y,m,b)
        # print("LOSS VALUE : ",loss_value)
        losses.append(loss_value)
        
    return m,b,losses,loss_value
#m,b,losses=gradient_descent(X,y,learning_rate,max_itr)
min_max_scaler=preprocessing.MinMaxScaler()
scaled_X=min_max_scaler.fit_transform(X)
# print(scaled_X)
scaled_y = min_max_scaler.fit_transform(np.reshape(y, (y.shape[0], 1)))
# print(scaled_y)
# print("Scaled X : ",scaled_X.shape)
# print("Scaled y : ",scaled_y.shape)
scaled_y=np.reshape(scaled_y,(scaled_y.shape[0]))
# print("Scaled y : ",scaled_y.shape)
m,b,losses,final_loss=gradient_descent(scaled_X,scaled_y,learning_rate,max_itr)
print("Slope(m)=",m)
print("Intercept(b)=",b)
print("final loss : ",final_loss)
plt.plot(losses)
plt.title('LOSSES',fontsize=18)
plt.xlabel('num of iteration ',fontsize=14)
plt.ylabel('loss value ',fontsize=14)
plt.show()

# plt.plot(X,y) #join lines
plt.title("Linear reg for Student data : Expected Marks vs Hours Studied",color='black',fontsize=18)
plt.xlabel('Hours Studied',color='red',fontsize=20)
plt.ylabel('Marks Predicted',color='red',fontsize=20)
plt.scatter(scaled_X,scaled_y,color='red')
y_pred=h(scaled_X,m,b)
plt.plot(scaled_X,y_pred)
plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import math
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
np.random.seed(310551094)
learning_rate = 0.6

def generate_linear(n = 100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1])  / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    import numpy as np
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground Truth', fontsize = 18)
    for i in range (x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.subplot(1, 2, 2)
    plt.title('Predict Result', fontsize = 18)
    for i in range (x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()




def sigmoid(A):
    return 1.0 / (1.0 + np.exp(-A))

def derivative_sigmoid(A):
    return sigmoid(A) * (1 - sigmoid(A)) 

def ReLU(A):
    A[A <= 0] = 0
    return A

def derivative_ReLU(A):
    A[A <= 0] = 0
    A[A > 0] = 1
    return A


def main(case):
    if(case != -1):
        n = case
        X, Y_hat = generate_linear(n)
        print("This is linear case:")
    else:
        n = 11
        X, Y_hat = generate_XOR_easy()
        print("This is XOR case:")

    loss_array = []
    epoch_array = []
    hidden_layer_num = 4
    W1 = np.random.randn(2, hidden_layer_num)
    W2 = np.random.randn(hidden_layer_num, hidden_layer_num)
    W3 = np.random.randn(hidden_layer_num, 1)
    
    #training
    for epoch in range(50000 + 1):
        #forward
        Z1 = np.matmul(X, W1)
        A1 = sigmoid(Z1)

        Z2 = np.matmul(A1, W2)
        A2 = sigmoid(Z2)

        Z3 = np.matmul(A2, W3)
        A3 = sigmoid(Z3)

        Y = A3
        diff = Y - Y_hat
        
        loss = np.mean(((Y - Y_hat) ** 2))/2
        if(epoch % 1000 == 0): 
            print("cur epoch num is :", epoch, ", cost:", loss)
        loss_array.append(loss)
        epoch_array.append(epoch)

        
        loss_d = (Y - Y_hat) / n

        #back propagation
        #gradient_a3 = loss_d
        gradient_3 = derivative_sigmoid(A3) * loss_d
        gradient_2 = derivative_sigmoid(A2) * np.matmul(gradient_3, W3.T)
        gradient_1 = derivative_sigmoid(A1) * np.matmul(gradient_2, W2.T)
        
        #updating
        W3 -= learning_rate * np.matmul(A2.T, gradient_3)
        W2 -= learning_rate * np.matmul(A1.T, gradient_2)
        W1 -= learning_rate * np.matmul(X.T, gradient_1)
    
    #testing       
    Z1 = np.matmul(X, W1)
    A1 = sigmoid(Z1)

    Z2 = np.matmul(A1, W2)
    A2 = sigmoid(Z2)

    Z3 = np.matmul(A2, W3)
    A3 = sigmoid(Z3)

    Y = A3
    diff = Y - Y_hat
    print(Y)
    
    
    miss = 0
    for x, y, y_hat in zip(X, Y, Y_hat):
        if((y < 0.5 and y_hat == 1) or (y > 0.5 and y_hat == 0)):
            miss = miss + 1
    '''
    for x, y in zip(X, Y_hat):
        if(y < 0.5):
            plt.scatter(x[0], x[1], c = 'g')
        else:
            plt.scatter(x[0], x[1], c = 'b')
    plt.title("ground")
    plt.show()
    
    for x, y in zip(X, Y):
        if(y < 0.5):
            plt.scatter(x[0], x[1], c = 'g')
        else:
            plt.scatter(x[0], x[1], c = 'b')
    plt.title("pred")
    plt.show()
    '''
    accuracy = 1 - miss / n 
    print(accuracy * 100, "%")
    
    if(case == -1):
        plt.title("XOR")
    else:
        plt.title("Linear")
    
    plt.plot(epoch_array, loss_array)
    plt.show()

    
    Y[Y <= 0.5] = 0
    show_result(X, Y, Y_hat)
    return W1, W2, W3


W1_trained_XOR, W2_trained_XOR, W3_trained_XOR = main(-1)
W1_trained_linear, W2_trained_linear, W3_trained_linear = main(30)


# In[2]:


print(W1_trained_XOR)
print(W2_trained_XOR)
print(W3_trained_XOR)


# In[3]:


print(W1_trained_linear)
print(W2_trained_linear)
print(W3_trained_linear)


# In[4]:


sig_x = np.linspace(-10,10,1001)
sig_y = sigmoid(sig_x)
der_sig_y = derivative_sigmoid(sig_x)
plt.plot(sig_x, sig_y)
plt.plot(sig_x, der_sig_y)
plt.show()


# In[ ]:





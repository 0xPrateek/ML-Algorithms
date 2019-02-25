import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def gradient_decent(x,y,theta0,theta1):
    m = len(x)
    i=0
    hypo = 0
    cost_func0 = 0
    cost_func1 = 0
    alpha = 0.01
    while i<m:
        hypo = theta0 + theta1*x[i]
        cost_func0 = cost_func0 + (hypo - y[i])
        cost_func1 = cost_func1 + (hypo - y[i])*x[i]
        i+=1

    cost_func0 = (1/m) * cost_func0
    cost_func1 = (1/m) * cost_func1
    theta0 = theta0 - alpha * cost_func0
    theta1 = theta1 - alpha * cost_func1

    return theta0,theta1

def cost_function(x,y,theta0,theta1):
    m = len(x)
    i=0
    hypo = 0
    mini = 0
    while i<m:
        hypo = theta0 + theta1 * x[i]
        mini = mini + np.power((hypo - y[i]),2)
        i+=1

    mini = mini/(2*m)
    return mini


if __name__ == '__main__':
    data = pd.read_csv('../../Datasets/linearRegression_Dataset.txt', header = None)
    x = data.iloc[:,0]
    y = data.iloc[:,1]
    theta0 = 0
    theta1 = 0
    i=0

    while(i<=1500):
    
        theta0,theta1 = gradient_decent(x,y,theta0,theta1)
        i+=1
        min_j = cost_function(x,y,theta0,theta1)
    
    print("Theta :: %.3f   %.3f " %(theta0,theta1))
    print("Cost value :: ",min_j)
    plt.scatter(x,y)
    plt.xlabel(" X axis")
    plt.ylabel(" Y axis")
    plt.plot(x,np.dot(x,theta1))
    plt.show()

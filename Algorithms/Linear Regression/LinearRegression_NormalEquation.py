import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
############################################################################################################################
'''
          Linear Regression implementation in Python 3
                    By - Prateek Mishra

 - Using Normal Equation for finding parameter values.
 - Without regularization.

:: Variables used ::
1) data :- This variable is used for storing data fetch from dataset file.
2) x,y  :- 'x' and 'y' is used for storing coloumns data of dataset file in seprate variables.
3) theta0, theta1 :- These are the parameters used in hypothesis function. (Initialized with 0)
4)  min_j :- It stores the value return by cost function after passing latest values of theta0 and theta1.
5) xt :- This is used for storing transpose of x.
6) m :- For counting number of training data.(Total number of rows)
7) xn,yn :- These are the scalar vector of used coloumns from datasets.
8) hypo :- For calculating hypothesis value (i.e h THETA (x) = theta0 + theta1*x).
12) k :- For counting number of training data which has been used for evaluation of Cost function.
13) mini :- This stores the value of cost function after minimization.

:: Functions used ::
1) normalequation() :- This fucntion is used for calculating the parameter valuese by using normal equation.
                       It returns two theta values(theta0,theta1) which is calculated.
2) cost_function() :- This function is used for calculating the cost function or squared mean error.

'''

def normalequation(x,y):

    xt = np.transpose(x)
    theta = np.linalg.inv(xt*x)*(np.transpose(x)*y)
    return theta

def cost_function(x,y,theta0,theta1):                   # This function is used for calculating Mean squared error or for minimization of cost function value.

    m = len(x)
    k=0
    hypo = 0
    mini = 0                                           # This will store the calculated minimized value of cost function.
    while k<m:                                         # calculating sumation of all the diffences between calculated hypothesis value and the actual yalue (i.e (h Theta (x) - y)^2)
        hypo = theta0 + theta1 * x[k]
        mini = mini + np.power((hypo - y[k]),2)
        k+=1

    mini = mini/(2*m)                                  # calculating average of the summed cost function value by dviding it with '2*m' and then returning the value.
    return mini


if __name__ == '__main__':

    data = pd.read_csv('/home/proton/Desktop/MachineLearning/Coursera/machine-learning-ex1/ex1/ex1data1.txt', header = None)

    xn = data.iloc[:,0]
    yn = data.iloc[:,1]
    x = np.transpose(np.matrix(data.iloc[:,0]))
    y = np.transpose(np.matrix(data.iloc[:,1]))
    x0 = np.transpose(np.matrix(np.ones(len(x))))
    x = np.hstack((x0,x))
    theta = normalequation(x,y)
    theta0 = theta[0,0]
    theta1 = theta[1,0]
    min_j = cost_function(xn,yn,theta0,theta1)
    print("Theta :: %.3f   %.3f " %(theta0,theta1))              # Displaying the values of theta which will be used for computation of hypothesis function.
    print("Cost value :: ",min_j)
    plt.scatter(np.array(x[:,1]),np.array(y))                    # Ploting graph of dataset and the hypothesis function.
    plt.xlabel(" X axis")
    plt.ylabel(" Y axis")
    plt.plot(x,np.dot(x,theta1))
    plt.show()

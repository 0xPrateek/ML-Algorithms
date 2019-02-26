############################################################################################################################
'''
          Linear Regression implementation in Python 3
                    By - Prateek Mishra

 - Using Gradient descent algorithm for finding parameter values.
 - Without regularization.

:: Variables used ::
1) data :- This variable is used for storing data fetch from dataset file.
2) x,y  :- 'x' and 'y' is used for storing coloumns data of dataset file in seprate variables.
3) theta0, theta1 :- These are the parameters used in hypothesis function. (Initialized with 0)
4) i :- For counting number of iterations done for calculating gradient decent and cost function.
5) min_j :- It stores the value return by cost function after passing latest values of theta0 and theta1.
6) m :- For counting number of training data.(Total number of rows)
7) j :- For counting number of training data which has been used for evaluation of gradient function.
8) hypo :- For calculating hypothesis value (i.e h THETA (x) = theta0 + theta1*x).
9) cost_func0 :- This is used to store cost function value for gradient descent of theta0.
10) cost_func1 :-This is used to store cost function value for gradient descent of theta1.
    (There are two cost function variables because they calculate different values. As given in
     the formaule of gradient decent for theat1 it ends up with product of 'x' which is not in
     case of calculating gradient descent for theat0)
11) alpha :- It is used for storing learning rate of gradient descent.
12) k :- For counting number of training data which has been used for evaluation of Cost function.
13) mini :- This stores the value of cost function after minimization.

:: Functions used ::
1) gradient_decent() :- This fucntion is used for calculating the parameter valuese by using gradient descent algorithm.
                       It returns two theta values(theta0,theta1) which is calculated.
2) cost_function() :- This function is used for calculating the cost function or squared mean error.

'''
##################################################################################################################################

import pandas as pd                                    # Importing required modules              
import matplotlib.pyplot as plt
import numpy as np

def gradient_decent(x,y,theta0,theta1):                # This fucntion will calculate parameter values using gradient descent algorithm.
    
    m = len(x)                                         # Initializing total number of training data.
    j=0                                                # Initializing counter for counting number of training data which has been used in calculating parameter values.         
    hypo = 0                                           # Initializing variable for storing hypothesis equation value. 
    cost_func0 = 0                                     # Initializing for storing cost function values.                     
    cost_func1 = 0
    alpha = 0.01                                       # Initializing learing rate for gradient descent algorithm.
    while j<m:                                         # finding sum of all the derivatives for calculating gradient descent.
        hypo = theta0 + theta1*x[j]
        cost_func0 = cost_func0 + (hypo - y[j])
        cost_func1 = cost_func1 + (hypo - y[j])*x[j]
        j+=1

    cost_func0 = (1/m) * cost_func0                    # Finding the average of the calculated derivatives by dviding it by 'm'
    cost_func1 = (1/m) * cost_func1
    theta0 = theta0 - alpha * cost_func0               # Finally calculating values of theta0 and theta1 and then returning it.
    theta1 = theta1 - alpha * cost_func1

    return theta0,theta1

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
    data = pd.read_csv('../../Datasets/linearRegression_Dataset.txt', header = None)    #Loading dataset file and storing in 'data' variable.
    x = data.iloc[:,0]           #sepratly storing the coloumn 0 in x.
    y = data.iloc[:,1]           #sepratly storing the coloumn 1 in y.
    theta0 = 0                   #Initializing theta values with '0'.
    theta1 = 0
    i=0                          #Initializing the iteration counter.
    while(i<=1500):              #Using iteration for finding the global minimum state and will consider the 1500'th iteration returning value as parameter value.

        theta0,theta1 = gradient_decent(x,y,theta0,theta1)        # Calling gradient_decent function which will return updatae theta values based on earlier values of theta.
        min_j = cost_function(x,y,theta0,theta1)                  # Calling cost_function for calculating squared mean error for the new value.
        i+=1

    print("Theta :: %.3f   %.3f " %(theta0,theta1))              # Displaying the values of theta which will be used for computation of hypothesis function.
    print("Cost value :: ",min_j)                                # Displaying the minimum cost function for final values of theta0 and theta1
    plt.scatter(x,y)                                             # Ploting graph of dataset and the hypothesis function.
    plt.xlabel(" X axis")
    plt.ylabel(" Y axis")
    plt.plot(x,np.dot(x,theta1))
    plt.show()

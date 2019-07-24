import pandas as pd
import numpy as np

class logistic:

    def __init__(self,x,y):
        self.data = x
        self.target = y
        self.theta = np.array([0,0,0,0])
        self.cost = 0
        self.thresh = 0.5

    def hypo(self,theta,X):

        z = (np.matrix(theta)*np.matrix(X).T)
        print("hypo : ",1/1+np.exp(-z))
        return 1/1+np.exp(-z)

    def gradient(self,m,alpha):

        n = 0
        while n<=1500:
            n=n+1
            self.theta = self.theta - (alpha/m)*(self.hypo(self.theta,self.data)-self.target)*np.matrix(self.data)
            print("Iteration :: ",n)
            print(self.theta)
            break

        print("Values of Theta :: ",self.theta)

    def predict(self,x):

        prediction = self.hypo(self.theta,x)
        print("The prediction probability is :: ",prediction)


if __name__  == "__main__":

    iris_data = pd.read_csv('/home/proton/Desktop/My/Might/MachineLearning/Python Implementation/Dataset/Iris.csv')
    iris_data=iris_data[:100]
    df = pd.DataFrame(iris_data)

    x = iris_data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    iris_data[['Species']] = np.where(df['Species']=='Iris-setosa',0,1)

    y = iris_data[['Species']]

    l = logistic(x,y)

    l.gradient(100,0.1)

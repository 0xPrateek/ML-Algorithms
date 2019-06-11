'''
    Algorithm Name :- Random Forest
    Decision Tree Split Algorithm :- Id3
    Author Nmae :- Prateek Mishra (0xPrateek)

'''

# Importing modules
import pandas as pd
from numpy import log2
import numpy,random

# Calculating entropy of single attribute
def single_attribute_entropy(data):
    entropy = 0
    target = data.columns[-1]
    attributes = data[target].unique()
    values = data[target].value_counts()
    for attribute in  attributes[::-1]:
        fraction = values[attribute]/len(data[target])
        entropy += - (fraction) * log2(fraction)

    return entropy

# Calculating entropy of the split
def double_attribute_entropy(data,attribute):
    entropy =0
    sub_attributes = data[attribute].unique()
    target_class = data.columns[-1]
    for sub_attribute in sub_attributes:
        entropy_sub = 0
        for target_clas in data[target_class].unique():
            values = data[attribute][data[attribute]==sub_attribute][data[target_class]==target_clas]
            if len(values)!=0:
                fraction = len(values)/len(data[attribute][data[attribute]==sub_attribute])
                entropy_sub += -(fraction) *log2(fraction)
            else:
                fraction = 0
        probab_attribute = len(data[attribute][data[attribute]==sub_attribute])/len(data[attribute])
        entropy += probab_attribute*entropy_sub

    return entropy


# This Function Returns the best split node
def get_node(data):
    entropy1 = single_attribute_entropy(data)
    IG = []
    attributes = data.columns[:-1]
    for attribute in attributes:
        entropy2 = double_attribute_entropy(data,attribute)
        IG.append(entropy1-entropy2)
    value = numpy.max(IG)
    index = IG.index(value)

    return attributes[index]

# This function return the Decision tree
def build_decision_tree(data,Tree = None):
    node = get_node(data)

    if Tree is None:
        Tree = {}
        Tree[node] = {}

    node_atbs = data[node].unique()
    for node_atb in node_atbs:
        subtable = data[data[node]==node_atb].reset_index(drop = True)
        values = subtable[subtable.columns[-1]].value_counts()
        if len(values) == 1:
            Tree[node][node_atb] = list(dict(values).keys())[0]
        else:
            Tree[node][node_atb] = build_decision_tree(subtable)

    return Tree

'''
 This function is for predicting the
 target value when given tree and input values.
'''
def predict(tree,values):
    for key in tree.keys():
        value = values[key]
        tree = tree[key][value]
        prediction = 0

        if type(tree) is dict:
            prediction = predict(tree,values)
        else:
            prediction = tree
            break;

    return prediction

if __name__ == "__main__":

    # Loading the dataset
    data = pd.read_csv('/home/proton/Desktop/tennis.csv')

    '''
     Some data preprocessing thing, Since
     the dataset used for the program is
     having some columns which doen't affect
     in prediction
    '''
    target = data['play']
    del data['day']

    # Total number of Decision trees in Random Forest
    n = 2

    # Minimum size of dataset
    k = 9

    a,b = data.shape

    '''
     Creating 'n' number of tress based on
     random dataset and then adding it into list
    '''
    trees = []
    for i in range(n):
        start = random.randint(0,a-k)
        stop = start + k
        training_data = data[start:stop]
        tree = build_decision_tree(training_data)
        trees.append(tree)

    # Taking user input for prediction by random forest
    print("[+] A forest with {} trees are created :) \n".format(n))
    print("\t:: Enter values for predicting ::\n")
    question = data.columns.tolist()
    values = {}
    for index,ques in enumerate(question[:-1]):
        values.update(dict({question[index]:input("Enter "+question[index]+" : ")}))

    '''
     Calling user defined function "predict()"
     for predicting based on various created
     trees and then storing it into a list
    '''
    predictions =[]
    for the_tree in trees:
        predictions.append(predict(the_tree,values))

    '''
     Checking for the most frequent prediction
     by the random forest which is the final
     prediction of the random forest.
    '''
    predicted,counts = numpy.unique(predictions,return_counts= True)
    if len(predicted) > 1:
        max_index = list(counts).index(numpy.max(counts))
        print("\nPrediction by random Forest : ",predicted[max_index])
    elif len(predicted) == 1:
        print("\nPrediction by random Forest : ",predicted[0])

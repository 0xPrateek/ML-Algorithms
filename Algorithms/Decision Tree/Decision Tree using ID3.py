'''
 Algorithm Name :- Decision Tree
 Best split Algorithm :- Id3
 Author Name :- Prateek Mishra (0xPrateek)
'''
import pandas as pd
from numpy import log2
import numpy
from pprint import pprint

def single_attribute_entropy(data):
    entropy = 0
    target = data.columns[-1]
    attributes = data[target].unique()
    values = data[target].value_counts()
    for attribute in  attributes[::-1]:
        fraction = values[attribute]/len(data[target])
        entropy += - (fraction) * log2(fraction)

    return entropy

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

def predict(tree,values):

    for key in tree.keys():

        value = values[key]
        tree = tree[key][value]
        prediction = 0

        if type(tree) is dict:
            predicted_value = predict(tree,values)
        else:
            predicted_value = tree
            break;

    return predicted_value

if __name__ == "__main__":

    data = pd.read_csv('/home/proton/Desktop/tennis.csv')
    print(data)
    target = data['play']
    del data['day']

    training_data = data[:10]
    testing_data = data[10:]

    # Building the decision tree
    print("[+] Building Decision tree ")
    tree = build_decision_tree(training_data)

    print("\tEnter values for predicting ::\n")
    question = data.columns.tolist()
    values = {}
    for index,ques in enumerate(question[:-1]):
        values.update(dict({question[index]:input("Enter "+question[index]+" : ")}))

    predicted = predict(tree,values)
    pprint(tree)
    print("The prediction is ::",predicted)

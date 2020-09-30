'''Implementation of Decision tree Algorithim in python using Entropy and Information Gain'''

#Importing packages
from math import log
import numpy as np
import operator
import json
from sklearn.model_selection import train_test_split
from sklearn import metrics
eps = np.finfo(float).eps
import pandas as pd

# convert origin data to list
def txtConvert():
    datalist = []
    with open('dt_data.txt', 'r') as f:
        lines = [line.rstrip() for line in f]
        lines[0] = lines[0][1:-1]
        for i in range(2, len(lines)):
            lines[i] = lines[i][4:-1]  # remove the original index and symbol ; in txt file
        del lines[1]  # delete the blank line
        for line in lines:
            line = line.split(', ')  # put each line into a list
            datalist.append(line)  # got completely listed data
    #df = pd.DataFrame(datalist[1:], columns=datalist[0])
    #df.to_csv("dt_data.csv")
    return datalist


# dictionary of label values, value:number
def labelVal(datalist):
    labelCounts = {}
    for sample in datalist:  # generate dic of labels
        labelvalue = sample[-1]  # label of current sample
        if labelvalue not in labelCounts.keys():
            labelCounts[labelvalue] = 0
        labelCounts[labelvalue] += 1
    return labelCounts


# compute Entropy of a data set
def dataEnt(datalist):
    num = len(datalist)  # number of sample
    labelCounts = labelVal(datalist)
    Entropy = 0.0  # initialize entropy
    for key in labelCounts:  # compute entropy
        prob = float(labelCounts[key]) / num
        Entropy = Entropy - prob * log(prob, 2)
    return Entropy


# sample set with particular attribute, feature, label
def featureSet(datalist, attribute, feature, label=None):
    newlist = []
    attri_index = completeAttri.index(attribute)
    if label == None:
        for s in datalist:
            if s[attri_index] == feature:
                newlist.append(s)
    else:
        for s in datalist:
            if s[attri_index] == feature and s[-1] == label:
                newlist.append(s)
    return newlist


# compute info_gain of each attribute in remaining attributes
def infoGain(datalist, attriList):
    target_variables = labelVal(datalist)

    # initialize attribute dictionary
    attriDic = {}
    for attribute in attriList:
        attriDic[attribute] = 0

    # compute info_gain for each attribute
    for attribute in attriList:
        i = attriList.index(attribute)
        info_gain_attribute = dataEnt(datalist)  # initialize info_gain of the attribute with data entropy

        # collect variables under this attribute
        variables = {}
        for j in range(len(datalist)):
            if datalist[j][i] not in variables.keys():
                variables[datalist[j][i]] = 1
            else:
                variables[datalist[j][i]] += 1

        # compute info_gain of the attribute
        for variable in variables.keys():
            entropy_each_feature = 0  # initialize feature entropy
            num_var = 0
            for target_variable in target_variables.keys():

                num = len(featureSet(datalist, attribute, variable, target_variable))  # numerator
                den = variables[variable]  # denominator
                num_var = den
                p = num / den
                if p == 0:
                    entropy_each_feature += -p * log(p + eps, 2)  # This calculates entropy for one feature value
                else:
                    entropy_each_feature += -p * log(p, 2)
            p2 = num_var / len(datalist)
            info_gain_attribute += -p2 * entropy_each_feature

        attriDic[attribute] = info_gain_attribute
    return attriDic


# to select the proper attribute in the current level in the tree
def bestAttri(datalist, attriList):
    dic = sorted(infoGain(datalist, attriList).items(), key=lambda x:x[1], reverse=True)  # load sorted attributes by infoGain
    best = list(dic)[0][0]
    return best


##################################### Below recursion to buil decision tree


# get the majority class label
def majorCnt(labelList):
    classCount = {}
    for i in labelList:
        if i not in classCount.keys():
            classCount[i] = 0
        classCount[i] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# to check whether all values of each attribute in each sample are the same to tell whether we need pruning
def attriSame(datalist):
    if len(datalist) > 1:
        newlist = []
        for s in datalist:
            newlist.append(s[:-1])  # remove label values and put into a new list
        return newlist[1:] == newlist[:-1]


# recursion to generate decision tree
def DecisionTree(datalist, attributes):
    labelList = [sample[-1] for sample in datalist]  # collect label of each sample into list
    if labelList.count(labelList[0]) == len(labelList):  # if the data is pure
        return labelList[0]
    if attriSame(datalist):
        return majorCnt(labelList)
    if len(attributes) == 1:  # return major attribute
        return majorCnt(labelList)

    chosenAttri = bestAttri(datalist, completeAttri)
    chosenAttri_index = completeAttri.index(chosenAttri)
    tree = {chosenAttri: {}}
    if chosenAttri in attributes:
        attributes.remove(chosenAttri)  # remove chosen attribute from the list of attribute
    featureValues = [example[chosenAttri_index] for example in datalist]
    values = set(featureValues)  # all possible values of this attribute
    for value in values:  # recursion
        subLabels = attributes[:]
        tree[chosenAttri][value] = DecisionTree(
            featureSet(datalist, chosenAttri, value), subLabels)
    return tree


data = txtConvert()[1:]
completeAttri = txtConvert()[0][:-1]
DT = DecisionTree(data, txtConvert()[0][:-1])
#print(DT)
#{'Occupied': {'High': {'Location': {'Mahane-Yehuda': 'Yes', 'German-Colony': 'No', 'City-Center': 'Yes', 'Talpiot': 'No'}}, 'Moderate': {'Location': {'German-Colony': {'VIP': {'Yes': 'Yes', 'No': 'No'}}, 'City-Center': 'Yes', 'Mahane-Yehuda': 'Yes', 'Talpiot': {'Price': {'Normal': 'Yes', 'Cheap': 'No'}}, 'Ein-Karem': 'Yes'}}, 'Low': {'Location': {'Ein-Karem': {'Price': {'Normal': 'No', 'Cheap': 'Yes'}}, 'Mahane-Yehuda': 'No', 'City-Center': {'Price': {'Normal': 'No', 'Cheap': 'No'}}, 'Talpiot': 'No'}}}}


##################################### Below plotting the decision tree


def PrintTree(DT):
    tree = DT
    tree_str = json.dumps(tree, indent=4)
    tree_str = tree_str.replace("\n    ", "\n")
    tree_str = tree_str.replace('"', "")
    tree_str = tree_str.replace(',', "")
    tree_str = tree_str.replace("{", "")
    tree_str = tree_str.replace("}", "")
    tree_str = tree_str.replace("    ", " | ")
    tree_str = tree_str.replace("  ", " ")
    print(tree_str)


PrintTree(DT)

##################################### Below prediction with new data

# to predict data with decision tree
def prediction(decisionTree, data):
    attri = list(decisionTree.keys())[0]  # the attribute of current node
    for v in decisionTree[attri].keys():
        if v == data[attri]:
            if isinstance(decisionTree[attri][v], dict):
                return(prediction(decisionTree[attri][v], data))
            else:
                return(decisionTree[attri][v])

##################################### Accuracy
def Accuracy():
    attribute=txtConvert() 
    for i in range(0,len(attribute)):
        attribute[i]=attribute[i][0:6]
    attribute = attribute[1:]
    label=txtConvert()
    for i in range(0,len(label)):
        label[i]=label[i][6:]
    label = label[1:]
    attribute_train, attribute_test, label_train, label_test = train_test_split(attribute,label,test_size=0.25, random_state=1)
    df=pd.DataFrame(attribute_test)
    label_pred=[] 
    for i in range(0,len(df)):
        test={'Occupied':df[0][i],'Price':df[1][i],'Music':df[2][i],'Location':df[3][i],'VIP':df[4][i],'Favourite Beer':df[5][i]}
        label_pred_i = prediction(DT, test)
        label_pred.append([label_pred_i])
    print("Accuracy of Decision Tree:",metrics.accuracy_score(label_test, label_pred))
Accuracy()

##################################### Predicting for the given test condition
data = {'Occupied':'Moderate', 'Price':'Cheap', 'Music':'Loud', 'Location':'City-Center', 'VIP':'No', 'Favorite Beer':'No'}
print("Prediction for the given test condition where (occupied = Moderate; price = Cheap; music = Loud; location = City-Center; VIP = No; favorite beer = No) Enjoy = ",prediction(DT, data))
'''
Created on Aug 8, 2018

@author: User
'''
from sklearn import tree
import numpy as np
import csv, datetime
from pprint import pprint


def transformTrainTitanicData(trainingFile, features):
    transformData = []
    # contains list of labels (Survived 0/1) for each of the passengers
    labels = []
    genderMap = {"male":1, "female":2, "":""}
    embarkMap = {"C":1, "Q":2, "S":3, "":""}
    # Initializing blank string to perform the check of a missing values
    blank = ""
    
    with open(trainingFile, 'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        lineNum = 1
        for row in lineReader:
            if lineNum == 1:
                header = row
            
            else:
                # list where categorical variables are converted to numerical ones
                allFeatures = list(map(lambda x:genderMap[x] if row.index(x) == 4
                                  else embarkMap[x] if row.index == 11 else x, row))
                featureVector = [allFeatures[header.index(feature)] for feature in features]
                if blank not in featureVector:
                    transformData.append(featureVector)
                    labels.append(int(row[1]))
            lineNum = lineNum + 1
        return transformData, labels


# Similar to above but tracks passengers ids instead of labels
# And needs to manage missing values
def transformTestTitanicData(testFile, features):
    transformData = []
    ids = []
    genderMap = {"male":1, "female":2, "":1}  # default gender is male
    embarkMap = {"C":1, "Q":2, "S":3, "":3}  # default port of embarcation is Southgampton
    # Initializing blank string to perform the check of a missing values
    blank = ""
    
    with open(testFile, 'r') as csvfile:
        lineReader = csv.reader(csvfile, delimiter=',', quotechar="\"")
        lineNum = 1
        for row in lineReader:
            if lineNum == 1:
                header = row
            else:
                # list where categorical variables are converted to numerical ones
                allFeatures = list(map(lambda x:genderMap[x] if row.index(x) == 3
                                  else embarkMap[x] if row.index == 10 else x, row))
                

#                 print(featureVector)
#                 featureVector = list(map(lambda x: 0 if x == "" else x, featureVector))
#                 
#                 # default Pclass is 3
                if allFeatures[1] == '':
                    allFeatures[1] = 3
#                 # Default Age is median age - 28 anos
                if allFeatures[4] == '':
                    allFeatures[4] = 28
#                 # Default number of companions is 0
                if allFeatures[5] == '':
                    allFeatures[5] = 0
#                 # Default fare - MEDIAN fare 14.45
                if allFeatures[8] == '':
                    allFeatures[8] = 15
                featureVector = [allFeatures[header.index(feature)] for feature in features]
                transformData.append(featureVector)
                ids.append(int(row[0]))
            lineNum = lineNum + 1
    return transformData, ids


# Takes classifier and runs it on a test data
# Universal function that can take it on any classifier
def titanicTest(classifier, resultFile, features):  # , transformDataFunction=transformTestTitanicData): #TBD
    testFile = r"C:\eclipse-workspace\Python_ML\TitanicSurvivor\data\test.csv"
    testData = transformTestTitanicData(testFile, features)
#     pprint(testData[0])
    result = classifier.predict(testData[0])
    with open(resultFile, "w", newline='') as f:
        ids = testData[1]
        lineWriter = csv.writer(f, delimiter=',', quotechar="\"")
        # Kaggle requires for submittion file to have a header
        lineWriter.writerow(["PassengerId", "Survived"])
        for rowNum in range(len(ids)):
            try:
                lineWriter.writerow([ids[rowNum], result[rowNum]])
            except Exception as e:
                print(e)

    
if __name__ == '__main__':
    trainingFile = r"C:\eclipse-workspace\Python_ML\TitanicSurvivor\data\train.csv"
    resultFile = r"C:\eclipse-workspace\Python_ML\TitanicSurvivor\data\TITANIC_DT_FIN_" + str(datetime.datetime.now().timestamp()).replace(".", "") + ".csv"
    features = ["Pclass", "Sex", "Age"]
    transformed_DS = transformTrainTitanicData(trainingFile, features)   

    X = np.array(transformed_DS[0])
    y = np.array(transformed_DS[1])
    
#     gini_decision_tree = tree.DecisionTreeClassifier(min_samples_split=100, max_leaf_nodes=15)
#     
#     gini_decision_tree = gini_decision_tree.fit(X, y)
#     
#     with open("titanic.dot", "w") as f:
#         f = tree.export_graphviz(gini_decision_tree, feature_names=features, class_names=["Dead", "Survived"],
#                                  filled=True, rounded=True, special_characters=True, out_file=f)
#     gini_importance = gini_decision_tree.feature_importances_
#     
#     #--------------------------------------------------
#     titanicTest(gini_decision_tree, resultFile, features)    
    #--------------------------------------------------
    
    entropy_decision_tree = tree.DecisionTreeClassifier(criterion="entropy", min_samples_split=100, max_leaf_nodes=15)
     
    entropy_decision_tree = entropy_decision_tree.fit(X, y)
     
    with open("entropy_titanic.dot", "w") as f:
        f = tree.export_graphviz(entropy_decision_tree, feature_names=features, class_names=["Dead", "Survived"],
                                 filled=True, rounded=True, special_characters=True, out_file=f)
    entropy_importance = entropy_decision_tree.feature_importances_
    print(entropy_importance)

    #--------------------------------------------------
    titanicTest(entropy_decision_tree, resultFile, features)    
    #--------------------------------------------------
    

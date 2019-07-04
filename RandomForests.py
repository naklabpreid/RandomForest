# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:05:43 2015

@author: sos
"""

import os
import pprint
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import precision_recall_fscore_support as prfs
# from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

baseDir = "../SVM/expAction/"
actionType = ['boxing', 'clap', 'punch']
extractDim = 2000
descriptorType = ["HOG", "HOF", "MBHx", "MBHy"]

# paramR = {'n_estimators':np.arange(500)}
paramG = {'n_estimators':[1, 10, 50, 100, 200, 400, 800, 1000]}

def readData(actionName, descType, mode):
    dir = baseDir + actionName + "/" + mode + "/" + descType
    fileNames = [filename for filename in os.listdir(dir) if not filename.startswith('.')]

    fileSize = len(fileNames)
    desc = [[0 for i in range(4000)] for j in range(fileSize)]

    count = 0
    for fileName in sorted(fileNames):
        data = np.genfromtxt(dir + "/" + fileName, delimiter=",")
        data = data[:data.shape[0] - 1]
        desc[count] = data
        count += 1

    return desc

def main():
    resultStr = []
    for descType in descriptorType:
        # read training data
        isFirst = True
        for action in actionType:
            actionDesc = readData(action, descType, "train")
            label = [action] * len(actionDesc)

            descArray = np.array(actionDesc)
            lebelArray = np.array(label)

            if isFirst:
                isFirst = False
                allTrainDesc = descArray
                allTrainLabel = lebelArray
            else:
                allTrainDesc = np.vstack((allTrainDesc, descArray))
                allTrainLabel = np.vstack((allTrainLabel, lebelArray))

        allTrainLabel = allTrainLabel.reshape(allTrainDesc.shape[0], 1)

        # training RFs
        rfc = GridSearchCV(estimator=RFC(random_state=0), param_grid=paramG)
        rfc.fit(allTrainDesc, allTrainLabel)

        # create sorted important index
        important_index = [i for importance, i in sorted(
            zip(rfc.best_estimator_.feature_importances_, range(rfc.best_estimator_.n_features_)),
            key=lambda x:x[0], reverse=False)]


        # re-training removing general action component for PReID
        dataNum = int(allTrainDesc.shape[0] / len(actionType))
        for action in actionType:
            # load label
            labelList = []
            labelFile = open(baseDir + action + "/label.txt", "r")
            lines = labelFile.readlines()
            for line in lines:
                text = line.replace('\n','')
                labelList.append(text)
            labelFile.close()

            # load desc
            actionIndex = actionType.index(action)
            descList = []
            for eachDesc in allTrainDesc[actionIndex*dataNum : actionIndex * dataNum + dataNum]:
                extractedDesc = eachDesc[important_index]
                # extractedDesc = extractedDesc[extractedDesc > 0.0]
                extractedDesc = extractedDesc[:extractDim]
                descList.append(extractedDesc)

            # learning SVM
            svc = svm.LinearSVC(C=0.03)
            svc.fit(descList, labelList)

            # load test data
            actionDesc = readData(action, descType, "test")
            dir = baseDir + action + "/test/" + descType 
            testNames = [filename for filename in os.listdir(dir) if not filename.startswith('.')]
            validCount = 0
            for fileName in sorted(testNames):
                data = np.genfromtxt(dir + "/" + fileName, delimiter=",")

                data = data[important_index]
                # data = data[data > 0.0]
                data = data[:extractDim]
                data = data.reshape(1, -1)

                pred = svc.predict(data)
                if pred[0] in fileName:
                    validCount += 1

            result = descType + " " + action + " " + str(extractDim) + " : " + str(validCount / len(testNames)) + "(" + str(rfc.best_estimator_.n_estimators) + ")"
            resultStr.append(result)

    for res in resultStr:
        print(res)

if __name__ == "__main__":
    main()
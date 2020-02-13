import os 
import json
from HMMTrigram import HmmTrigram 
from sklearn.metrics import confusion_matrix
import numpy as np


def LoadRawData(path):
    print("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    
    lines = f.readlines()

    return lines

def LoadJsonTokens(xTrain , Tokenize = True):
    parsedJsonTokens = []
    for sentence in xTrain:
         lineJsonTokens = json.loads(sentence)
         if(Tokenize == True):
             lineJsonTokens.insert(0,["<sw>","<st>"])
             lineJsonTokens.insert(0,["<sw>","<st>"])
             lineJsonTokens.append(["</sw>","</st>"])
         parsedJsonTokens.append(lineJsonTokens)
    return parsedJsonTokens;

def LoadJsonTokensTest(xTrain):
    parsedJsonTokens = []
    for sentence in xTrain:
         lineJsonTokens = json.loads(sentence)
         parsedJsonTokens.append(lineJsonTokens)
    return parsedJsonTokens;

def CalculateAccuracy(model, predictedTags , xDevParsed):
    correctCount = 0
    totalCount = 0
    actual = []
    predicted = []
    for i in range(len(xDevParsed)):
        for j in range(1,len(xDevParsed[i])-1):
            totalCount = totalCount + 1
            actual.append(xDevParsed[i][j][1])
            predicted.append(predictedTags[i][j])
            if( xDevParsed[i][j][1] == predictedTags[i][j]):
                correctCount = correctCount + 1
    return (correctCount/totalCount , confusion_matrix(actual,predicted, labels=model.AllTags))
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

xTrainRaw = LoadRawData("twt.bonus.json")
xTrainParsed = LoadJsonTokens(xTrainRaw)

param = {}
param["oovfrequency"] = 1
param["usePartialTrainingData"]  = False
param["lambda2"]  = 0.15
param["lambda1"]  = 0.8
param["lambda3"]  = 0.05
param["smoothingfactor"] = .0001
param["debug"] = True
if( param["usePartialTrainingData"] == True):
    xTrainParsed = xTrainParsed[:100]

hmmTrigram = HmmTrigram(xTrainParsed, param)

xDevRaw = LoadRawData("twt.test.json")
xDevParsed = LoadJsonTokensTest(xDevRaw)
if( param["usePartialTrainingData"] == True):
  xDevParsed = xDevParsed[:10]

predictedtags = hmmTrigram.FindBestTagSequences(xDevParsed, param)
accuracy,confusionMatrix  = CalculateAccuracy(hmmTrigram, predictedtags, xDevParsed)
allTags = hmmTrigram.AllTags.copy()

print("Accuracy : ", accuracy)
print_cm(confusionMatrix, allTags)


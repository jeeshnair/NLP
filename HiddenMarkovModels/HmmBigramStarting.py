import os 
import json
from HmmModels import HmmBigram 

def LoadRawData(path):
    print("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    
    lines = f.readlines()

    return lines

def LoadJsonTokens(xTrain):
    parsedJsonTokens = []
    for sentence in xTrain:
         lineJsonTokens = json.loads(sentence)
         lineJsonTokens.insert(0,["<sw>","<st>"])
         lineJsonTokens.append(["</sw>","</st>"])
         parsedJsonTokens.append(lineJsonTokens)
    return parsedJsonTokens;

def CalculateAccuracy(predictedTags , xDevParsed):
    correctCount = 0
    totalCount = 0
    for i in range(len(xDevParsed)):
        for j in range(1,len(xDevParsed[i])-1):
            totalCount = totalCount + 1
            if( xDevParsed[i][j][1] == predictedTags[i][j][1]):
                correctCount = correctCount + 1
    return correctCount/totalCount;

xTrainRaw = LoadRawData("C:\\source\\nlp\\HiddenMarkovModels\\TestData\\twt.train.json")
xTrainParsed = LoadJsonTokens(xTrainRaw)

param = {}
param["oovfrequency"] = 1
param["usePartialTrainingData"]  = False
param["lambda2"]  = 0.7
param["lambda1"]  = 0.3
param["smoothingfactor"] = .0001
param["debug"] = True
if( param["usePartialTrainingData"] == True):
    xTrainParsed = xTrainParsed[:100]

hmmBigram = HmmBigram(xTrainParsed, param)

xDevRaw = LoadRawData("C:\\source\\nlp\\HiddenMarkovModels\\TestData\\twt.dev.json")
xDevParsed = LoadJsonTokens(xDevRaw)
if( param["usePartialTrainingData"] == True):
    xDevParsed = xDevParsed[:10]

predictedtags = hmmBigram.FindBestTagSequences(xDevParsed, param)
accuracy  = CalculateAccuracy(predictedtags, xDevParsed)
print("Accuracy : ", accuracy)



 
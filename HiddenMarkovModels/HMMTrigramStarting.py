import os 
import json
from HMMTrigram import HmmTrigram 

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
param["lambda2"]  = 0.2
param["lambda1"]  = 0.6
param["lambda3"]  = 0.2
param["smoothingfactor"] = .0001
param["debug"] = True
if( param["usePartialTrainingData"] == True):
    xTrainParsed = xTrainParsed[:100]

hmmTrigram = HmmTrigram(xTrainParsed, param)

xDevRaw = LoadRawData("C:\\source\\nlp\\HiddenMarkovModels\\TestData\\twt.dev.json")
xDevParsed = LoadJsonTokensTest(xDevRaw)
if( param["usePartialTrainingData"] == True):
    xDevParsed = xDevParsed[:10]

predictedtags = hmmTrigram.FindBestTagSequences(xDevParsed, param)
#accuracy  = CalculateAccuracy(predictedtags, xDevParsed)
print("Accuracy : ", accuracy)



 

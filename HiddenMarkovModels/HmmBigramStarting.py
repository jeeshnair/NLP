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

xTrainRaw = LoadRawData("C:\\source\\nlp\\HiddenMarkovModels\\TestData\\twt.train.json")
xTrainParsed = LoadJsonTokens(xTrainRaw)

param = {}
param["oovfrequency"] = 1
param["useHalfTrainingData"]  = False
param["lambda2"]  = 0.5
param["lambda1"]  = 0.5
param["smoothingfactor"] = 1
param["debug"] = True
if( param["useHalfTrainingData"] == True):
    xTrainParsed = xTrainParsed[:100]


hmmBigram = HmmBigram(xTrainParsed, param)

xDevRaw = LoadRawData("C:\\source\\nlp\\HiddenMarkovModels\\TestData\\twt.dev.json")
xDevParsed = LoadJsonTokens(xDevRaw)

hmmBigram.FindBestTagSequences(xDevParsed, param)

 
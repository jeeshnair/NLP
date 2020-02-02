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

hmmBigram = HmmBigram(xTrainParsed, param)
 
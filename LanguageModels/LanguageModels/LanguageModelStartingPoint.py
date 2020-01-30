from LanguageModel import UnigramModel, BigramModel, TrigramModel
import os

def LoadRawData(path):
    print("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    
    lines = f.readlines()

    return lines

def TokenizeBigramDev(xDev):
    i = 0
    tokenizedTrain = []
    for sentence in xDev:
        sentence = sentence.rstrip()
        sentence = "<s>" + " " + sentence
        sentence = sentence + " " + "</s>"

        tokenizedTrain.append(sentence)  

    return tokenizedTrain

def TokenizeTrigramDev(xDev):
    i = 0
    tokenizedTrain = []
    for sentence in xDev:
        sentence = sentence.rstrip()
        sentence = "<s>" + " " + "<s>" + " " + sentence
        sentence = sentence + " " + "</s>"

        tokenizedTrain.append(sentence)  

    return tokenizedTrain

# All hyper parameters relevant to this model
param = {}
param["oovfrequency"] = 5 #Words equalto or below this frequency will not be part of vocabulary
param["smoothingFactor"] = .1 # Smoothing factor
param["uselinearInterpolation"] = True # Whether to use linear interpolation . This is mutually exclusive with smoothing
param["lambda3"] = 0.4 # Various lamba values for linear interpolation
param["lambda2"] = 0.35
param["lambda1"] = 0.25
param["useHalfTrainingData"] = False # Whether to use half or full training data for training.

print("hyperparameters", param)

xTrainRaw = LoadRawData("TestData/brown.train.txt")
xDevRaw = LoadRawData("TestData/brown.test.txt")
unigramModel = UnigramModel(xTrainRaw,param)
unigramPerplexity = unigramModel.CalculatePerplexity(xDevRaw , param)
print("Unigram perplexity",unigramPerplexity)
print("done with unigram")
s
xTrainRaw = LoadRawData("TestData/brown.train.txt")
xDevRaw = LoadRawData("TestData/brown.test.txt")
xDevRaw = TokenizeBigramDev(xDevRaw)
bigramModel = BigramModel(xTrainRaw,param)
bigramPerplexity = bigramModel.CalculatePerplexity(xDevRaw , param)
print("Bigram perplexity",bigramPerplexity)
print("done with bigram")

xTrainRaw = LoadRawData("TestData/brown.train.txt")
if( param["useHalfTrainingData"] == True):
    xTrainRaw = xTrainRaw[:int(len(xTrainRaw)/2)]
xDevRaw = LoadRawData("TestData/brown.test.txt")
xDevRaw = TokenizeTrigramDev(xDevRaw)
trigramModel = TrigramModel(xTrainRaw,param)
trigramPerplexity = trigramModel.CalculatePerplexity(xDevRaw , param)
print("Trigram perplexity",trigramPerplexity)
print("done with trigram")





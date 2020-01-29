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

param = {}
param["oovfrequency"] = 5
param["smoothingFactor"] = .1
param["uselinearInterpolation"] = False
param["lambda3"] = 0.4
param["lambda2"] = 0.35
param["lambda1"] = 0.25

print("hyperparameters", param)

xTrainRaw = LoadRawData("TestData/brown.train.txt")
xDevRaw = LoadRawData("TestData/brown.test.txt")
unigramModel = UnigramModel(xTrainRaw,param)
unigramPerplexity = unigramModel.CalculatePerplexity(xDevRaw , param)
print("perplexity",unigramPerplexity)
print("done with unigram")

xTrainRaw = LoadRawData("TestData/brown.train.txt")
xDevRaw = LoadRawData("TestData/brown.test.txt")
xDevRaw = TokenizeBigramDev(xDevRaw)
bigramModel = BigramModel(xTrainRaw,param)
bigramPerplexity = bigramModel.CalculatePerplexity(xDevRaw , param)
print("perplexity",bigramPerplexity)
print("done with bigram")

#xTrainRaw = LoadRawData("TestData/brown.train.txt")
#xDevRaw = LoadRawData("TestData/brown.train.txt")
#xDevRaw = TokenizeTrigramDev(xDevRaw)
#trigramModel = TrigramModel(xTrainRaw,param)
#trigramPerplexity = trigramModel.CalculatePerplexity(xDevRaw , param)
#print("perplexity",trigramPerplexity)
#print("done with trigram")

#xTrainRaw = LoadRawData("TestData/brown.train.txt")
#xDevRaw = LoadRawData("TestData/brown.dev.txt")
#xDevRaw = TokenizeTrigramDev(xDevRaw)
#trigramModel = TrigramModel(xTrainRaw,param)
#trigramPerplexity = trigramModel.CalculatePerplexity(xDevRaw , param)
#print("perplexity",trigramPerplexity)
#print("done with trigram")

#xTrainRaw = LoadRawData("TestData/brown.train.txt")
#xDevRaw = LoadRawData("TestData/brown.test.txt")
#xDevRaw = TokenizeTrigramDev(xDevRaw)
#trigramModel = TrigramModel(xTrainRaw,param)
#trigramPerplexity = trigramModel.CalculatePerplexity(xDevRaw , param)
#print("perplexity",trigramPerplexity)
#print("done with trigram")

xTrainRaw = LoadRawData("TestData/brown.train.txt")
#xTrainRaw = xTrainRaw[:int(len(xTrainRaw)/2)]
xDevRaw = LoadRawData("TestData/brown.test.txt")
xDevRaw = TokenizeTrigramDev(xDevRaw)
trigramModel = TrigramModel(xTrainRaw,param)
trigramPerplexity = trigramModel.CalculatePerplexity(xDevRaw , param)
print("perplexity",trigramPerplexity)
print("done with trigram")





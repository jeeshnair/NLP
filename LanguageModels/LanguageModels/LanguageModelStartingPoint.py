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
param["oovfrequency"] = 4
param["smoothingFactor"] = .05
param["uselinearInterpolation"] = True
xTrainRaw = LoadRawData("TestData/brown.train.txt")
xDevRaw = LoadRawData("TestData/brown.dev.txt")

#xTrainRaw = LoadRawData("TestData/brown.dev.txt")
#xDevRaw = xTrainRaw

# xTrainRaw = LoadRawData("TestData/small.txt")
# xDevRaw = LoadRawData("TestData/small.test.txt")
#unigramModel = UnigramModel(xTrainRaw,param)
#unigramPerplexity = unigramModel.CalculatePerplexity(xDevRaw , param)
#print("perplexity",unigramPerplexity)
#print("done with unigram")

xDevRaw = TokenizeBigramDev(xDevRaw)
bigramModel = BigramModel(xTrainRaw,param)
bigramPerplexity = bigramModel.CalculatePerplexity(xDevRaw , param)
print("perplexity",bigramPerplexity)
print("done with bigram")

#xDevRaw = TokenizeTrigramDev(xDevRaw)
#trigramModel = TrigramModel(xTrainRaw,param)
#trigramPerplexity = trigramModel.CalculatePerplexity(xDevRaw , param)
#print("perplexity",trigramPerplexity)
#print("done with trigram")



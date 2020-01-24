from LanguageModel import UnigramModel, BigramModel, TrigramModel
import os

def LoadRawData(path):
    print("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    
    lines = f.readlines()

    return lines;

param = {}
param["oovfrequency"] = 0
#xTrainRaw = LoadRawData("TestData/brown.train.txt")
#xDevRaw = LoadRawData("TestData/brown.dev.txt")

#xTrainRaw = LoadRawData("TestData/brown.dev.txt")
#xDevRaw = xTrainRaw

xTrainRaw = LoadRawData("TestData/small.txt")
xDevRaw = LoadRawData("TestData/small.test.txt")

unigramModel = UnigramModel(xTrainRaw,param)
unigramPerplexity = unigramModel.CalculatePerplexity(xDevRaw)
print("perplexity",unigramPerplexity)
print("done with unigram")

bigramModel = BigramModel(xTrainRaw,param)
bigramPerplexity = bigramModel.CalculatePerplexity(xDevRaw)
print("perplexity",bigramPerplexity)
print("done with bigram")

trigramModel = TrigramModel(xTrainRaw,param)
trigramPerplexity = trigramModel.CalculatePerplexity(xDevRaw)
print("perplexity",trigramPerplexity)
print("done with trigram")



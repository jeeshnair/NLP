from LanguageModel import UnigramModel, BigramModel, TrigramModel
import os

def LoadRawData(path):
    print("Loading data from: %s" % os.path.abspath(path))
    f = open(path, 'r')
    
    lines = f.readlines()

    return lines;

param = {}
param["oovfrequency"] = 1
xTrainRaw = LoadRawData("TestData/brown.train.txt")
xDevRaw = LoadRawData("TestData/brown.dev.txt")

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



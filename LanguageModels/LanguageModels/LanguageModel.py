import math
import random

class UnigramModel(object):
    """A unigram language model"""
    
    def __init__(self, xTrain , param):
        self.totalTrainWordCount = 0
        self.unigramTrainFrequency = dict()
        self.wordFrequency = dict()
        self.actualTrainWordCount = 0
        # used for unseen words in training vocabularies
        self.UNK = "unk"

        # sentence start and end
        self.SENTENCE_START = "<s>"
        self.SENTENCE_END = "</s>"
        self.vocabulary = []

        self.testwordCount = 0
        self.testBigramCount = 0
        self.testTrigramCount = 0

        self.CountWordsTrain(xTrain, param)
        self.InitializeVocabulary(param)
        xTrain = self.TokenizeTrain(xTrain)
        self.NormalizeUnigramTrain(xTrain,param)

        pass

    def InitializeVocabulary(self, param):
        unkFrequency = 0
        wordsSkipped = 0
        wordsToRemoveFromUnigramFrequency = []
        for word in self.wordFrequency:
            if self.wordFrequency[word] <= param["oovfrequency"]: # and wordsSkipped <= 500:
                wordsSkipped = wordsSkipped + 1
            elif(word not  in self.vocabulary):
                self.vocabulary.append(word)

    def CalculateUnigramSentenceProbability(self, sentence , param):
        words = sentence.split()
        sentenceLogProbability = 0 
        for word in words:
            sentenceLogProbability = sentenceLogProbability + self.CalculateUnigramWordProbability(word , param)

        return sentenceLogProbability

    def CalculateUnigramWordProbability(self , word , param):
        numerator = self.unigramTrainFrequency.get(self.UNK,0)
        denominator = self.totalTrainWordCount
        if(word in self.vocabulary):
            numerator = self.unigramTrainFrequency[word]

        numerator = numerator + param["smoothingFactor"]
        denominator = denominator + (param["smoothingFactor"] * len(self.unigramTrainFrequency))
        #print("probability of ",word,numerator / denominator)
        return math.log(numerator / denominator,2)


    def CalculatePerplexity(self, xDev , param):
        corpusLogProbability = 0
        self.CountWordsTest(xDev)
        for sentence in xDev:
            corpusLogProbability = corpusLogProbability + self.CalculateUnigramSentenceProbability(sentence , param)

        print("corpus log probability: ", corpusLogProbability)
        print("unigram count: ", self.testwordCount)

        return math.pow(2, -(corpusLogProbability / self.testwordCount))

    def CountWordsTrain(self, xTrainRaw,param):
        for sentence in xTrainRaw:
            words = sentence.split()
            for word in words:
                self.wordFrequency[word] = self.wordFrequency.get(word, 0) + 1
                self.totalTrainWordCount = self.totalTrainWordCount + 1

    def CountWordsTest(self, xDev):
        for sentence in xDev:
            wordCount = 0
            words = sentence.split()
            for word in words:
                wordCount = wordCount + 1
            self.testwordCount = self.testwordCount + wordCount

        print("wordcount test" , self.testwordCount)
        
    def TokenizeTrain(self, xTrainRaw):
        i = 0
        tokenizedTrain = []
        for sentence in xTrainRaw:
            sentence = sentence.rstrip()
            for word in sentence.split():
                if word not in self.vocabulary:
                    sentence = sentence.replace(" " + word + " ", " " + self.UNK + " ")
            tokenizedTrain.append(sentence)  

        return tokenizedTrain

    def NormalizeUnigramTrain(self, xTrainRaw,param):
        i = 0
        for sentence in xTrainRaw:
            words = sentence.split()
            for word in words:
                if (word != self.SENTENCE_END) and (word!=self.SENTENCE_START):
                    self.unigramTrainFrequency[word] = self.unigramTrainFrequency.get(word, 0) + 1
            i = i + 1
        print("wordcount train" , self.totalTrainWordCount)

class BigramModel(UnigramModel):
    """A unigram language model"""
    
    def __init__(self, xTrain, param , FromTrigram= False):
        self.bigramTrainFrequency = dict()
        self.bigramTrainCount = 0
        self.wordFrequency = dict()
        self.totalTrainWordCount = 0
        self.vocabulary = []
        self.testwordCount = 0
        self.unigramTrainFrequency = dict()

        self.CountWordsTrain(xTrain, param)
        self.InitializeVocabulary(param)
        xTrain = self.TokenizeBigramTrain(xTrain)
        self.NormalizeBigramTrain(xTrain,param)

    def CalculatePerplexity(self, xDev , param):
        corpusLogProbability = 0
        self.CountWordsTest(xDev)
        for sentence in xDev:
            corpusLogProbability = corpusLogProbability + self.CalculateBigramSentenceProbability(sentence, param)
        print("corpus log probability: ", corpusLogProbability)

        return math.pow(2, -(corpusLogProbability / self.testwordCount))

    def CalculateBigramSentenceProbability(self, sentence, param):
        words = sentence.split()
        sentenceLogProbability = 0 
        for i in range(len(words)):
            activeWord = words[i]
            if(i > 0):
                conditionWord = words[i - 1]
                sentenceLogProbability = sentenceLogProbability + self.CalculateBigramWordProbability(activeWord,conditionWord, param)

        return sentenceLogProbability

    def CalculateBigramWordProbability(self , activeWord, conditionWord, param):
        frequencyLookupKey = conditionWord + "_" + activeWord
        numerator = self.bigramTrainFrequency.get(frequencyLookupKey,0) + param["smoothingFactor"]
        denominator = self.unigramTrainFrequency.get(conditionWord,0) + (param["smoothingFactor"] * len(self.unigramTrainFrequency))

        #print("probability of ",frequencyLookupKey,numerator / denominator)

        return  math.log(numerator / denominator,2)

    def NormalizeBigramTrain(self, xTrainRaw,param):
        i = 0
        for sentence in xTrainRaw:
            words = sentence.split()
            for i in range(len(words)):
                self.unigramTrainFrequency[words[i]] = self.unigramTrainFrequency.get(words[i], 0) + 1
                if (i > 0):
                    self.bigramTrainCount = self.bigramTrainCount + 1
                    key = words[i - 1] + "_" + words[i]
                    self.bigramTrainFrequency[key] = self.bigramTrainFrequency.get(key,0) + 1
        print("bigram train" , self.bigramTrainCount)

    def TokenizeBigramTrain(self, xTrainRaw, FromTrigram = False):
        i = 0
        tokenizedTrain = []
        for sentence in xTrainRaw:
            sentence = sentence.rstrip()
            for word in sentence.split():
                if word not in self.vocabulary:
                    sentence = sentence.replace(" " + word + " ", " " + "unk" + " ")
            sentence = "<s>" + " " + sentence
            if(FromTrigram == True):
                sentence = self.SENTENCE_START + " " + sentence
            sentence = sentence + " " + "</s>"

            tokenizedTrain.append(sentence)  

        return tokenizedTrain

class TrigramModel(BigramModel):
    """A unigram language model"""
    
    def __init__(self, xTrain, param):
        self.trigramFrequency = dict()
        self.bigramTrainFrequency = dict()
        self.bigramTrainCount  = 0
        self.trigramTrainCount = 0
        self.wordFrequency = dict()
        self.totalTrainWordCount = 0
        self.vocabulary = []
        self.testwordCount = 0
        self.testBigramCount = 0
        self.unigramTrainFrequency = dict()

        self.CountWordsTrain(xTrain, param)
        self.InitializeVocabulary(param)
        xTrain = self.TokenizeTrigramTrain(xTrain)
        self.NormalizeTrigramTrain(xTrain,param)

    def CalculatePerplexity(self, xDev , param):

        corpusLogProbability = 0
        wordCount = self.CountWordsTest(xDev)
        for sentence in xDev:
            corpusLogProbability = corpusLogProbability + self.CalculateTrigramSentenceProbability(sentence, param)

        print("corpus log probability: ", corpusLogProbability)
        return math.pow(2, -(corpusLogProbability / self.testwordCount))

    def CalculateTrigramSentenceProbability(self, sentence , param):
        words = sentence.split()
        sentenceLogProbability = 0 
        for i in range(len(words)):
            if(i > 1):
                activeWord = words[i]
                conditionWord1 = words[i - 1]
                conditionWord2 = words[i - 2]
                sentenceLogProbability = sentenceLogProbability + self.CalculateTrigramWordProbability(activeWord,conditionWord1,conditionWord2, param)

        return sentenceLogProbability

    def CalculateTrigramWordProbability(self , activeWord, conditionWord1, conditionWord2 , param):
        triGramfrequencyLookupKey = conditionWord2 + "_" + conditionWord1 + "_" + activeWord
        biGramfrequencyLookupKey = conditionWord2 + "_" + conditionWord1
        numerator = self.trigramFrequency.get(triGramfrequencyLookupKey,0) + param["smoothingFactor"]
        denominator = self.bigramTrainFrequency.get(biGramfrequencyLookupKey,0) + (len(self.unigramTrainFrequency) * + param["smoothingFactor"])

        #print("probability of
        #",triGramfrequencyLookupKey,numerator/denominator)

        return  math.log(numerator / denominator,2)

    def NormalizeTrigramTrain(self, xTrainRaw,param):
        i = 0
        for sentence in xTrainRaw:
            words = sentence.split()
            for i in range(len(words)):
                self.unigramTrainFrequency[words[i]] = self.unigramTrainFrequency.get(words[i], 0) + 1
                if(i>0):
                    self.bigramTrainCount = self.bigramTrainCount + 1
                    key = words[i - 1] + "_" + words[i]
                    self.bigramTrainFrequency[key] = self.bigramTrainFrequency.get(key,0) + 1
                if (i > 1):
                    self.trigramTrainCount = self.trigramTrainCount + 1
                    key = words[i - 2] + "_" + words[i - 1] + "_" + words[i]
                    self.trigramFrequency[key] = self.trigramFrequency.get(key,0) + 1
        print("trigram train" , self.trigramTrainCount)

    def TokenizeTrigramTrain(self, xTrainRaw):
        i = 0
        tokenizedTrain = []
        for sentence in xTrainRaw:
            sentence = sentence.rstrip()
            for word in sentence.split():
                if word not in self.vocabulary:
                    sentence = sentence.replace(" " + word + " ", " " + "unk" + " ")
            sentence = "<s>" + " " + "<s>" + " " + sentence
            sentence = sentence + " " + "</s>"

            tokenizedTrain.append(sentence)  

        return tokenizedTrain

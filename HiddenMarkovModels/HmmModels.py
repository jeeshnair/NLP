import math

class HmmBigram(object):
    """A Hmm bigram language model"""
    
    def __init__(self, xTrain , param):
        self.xTrain = xTrain
        self.TagCount = dict()
        self.TransitionCount = dict()
        self.EmissionCount = dict()
        self.TotalTrainWordCount = 0
        self.Vocabulary = []
        self.LowFrequencyWords = []
        self.WordFrequency = dict()
        if(param["debug"]  == True):
            print("Initializing Vocabulary")
        self.InitializeVocabulary(xTrain,param)
        if(param["debug"]  == True):
            print("Learning the model")
        self.CalculateCounts(xTrain,param)
        self.AllTags = list(self.TagCount.keys())
        pass

    def InitializeVocabulary(self,xTrain,param):
        self.CountWordsTrain(xTrain, param)
        print("word count"," ",len(self.WordFrequency))
        for word in self.WordFrequency:
            if self.WordFrequency[word] > param["oovfrequency"]:
                if(word not in self.Vocabulary):
                    self.Vocabulary.append(word)
            #else:
            #    if(word not in self.Vocabulary):
            #        self.Vocabulary.append(word)
        
    def CountWordsTrain(self, xTrain,param):
        for tokens in xTrain:
           for i in range(len(tokens)):
                self.WordFrequency[tokens[i][0]] = self.WordFrequency.get(tokens[i][0], 0) + 1

    def TranslateToWordClass(self,word):
        numDigits = 0
        containsSlash = False
        containsAlpha = False
        containsComma = False
        containsPeriod = False
        containsDash = False
        containsDigits = False

        #for char in word:
        #    if char.isdigit():
        #        numDigits += 1
        #        containsDigits = True
        #    if char.isalpha():
        #        containsAlpha = True
        #    if char == ",":
        #        containsComma = True
        #    if char == "/":
        #        containsSlash = True
        #    if char == "-":
        #        containsDash = True
        #    if char == ".":
        #        containsPeriod = True   

        #if(word.isalnum() and containsDigits == True):
        #    return "containsDigitAndAlpha"
        #if word.isdigit() and len(word) == 2:
        #    return 'twoDigitNum'
        #if word.isdigit() and len(word) == 4:
        #    return 'fourDigitNum'
        #if word.isdigit(): #Is a digit
        #    return 'othernum'
        #if(containsDigits ==True and containsSlash == True and containsAlpha == False):
        #    return "containsDigitAndSlash";
        #if(containsDigits ==True and containsDash == True and containsAlpha == False):
        #    return "containsDigitAndDash";
        #if(containsDigits ==True and containsComma == True and containsAlpha == False):
        #    return "containsDigitAndDash";
        #if(containsDigits ==True and containsPeriod == True and containsAlpha == False):
        #    return "containsDigitAndPeriod";
        #elif digitFraction > 0.5:
        #    return 'mostlyDigits'
        #if word.islower(): #All lower case
        #    return 'lowercase'
        #elif word.isupper(): #All upper case
        #    return 'allCaps'
        #if word[0].isupper(): #is a title, initial char upper, then all lower
        #    return 'initCap'
        if word.startswith("@"): #is a title, initial char upper, then all lower
            return 'twitterHandle'
        elif word.startswith("#"):
            return 'hashTag'
        elif word.startswith("https"):
            return 'uri'
        #elif numDigits > 0:
        #    return 'containsDigits'  
        else:
            return "other"

    def CalculateCounts(self, xTrain, param):
        for tokens in xTrain:
            for i in range(len(tokens)):
                if(tokens[i][0] not in self.Vocabulary):
                    tokens[i][0] = "unk" #self.TranslateToWordClass(tokens[i][0])
                self.TotalTrainWordCount = self.TotalTrainWordCount + 1
                self.TagCount[tokens[i][1]] = self.TagCount.get(tokens[i][1],0) + 1
                self.EmissionCount[(tokens[i][1],tokens[i][0])] = self.EmissionCount.get((tokens[i][1],tokens[i][0]),0) + 1
                if(i > 0):
                    self.TransitionCount[(tokens[i - 1][1],tokens[i][1])] = self.TransitionCount.get((tokens[i - 1][1],tokens[i][1]),0) + 1

    def GetTransitionProbability(self, conditionState, currentState , param):
        probabilityWithLinearInterpolation = 0

        #bigram part
        numerator = self.TransitionCount.get((conditionState, currentState),0) 
        denominator = self.TagCount.get(conditionState,0)
        probabilityWithLinearInterpolation = probabilityWithLinearInterpolation + (param["lambda1"] * (numerator / denominator))

        #unigram part
        numerator = self.TagCount.get(currentState) 
        denominator = self.TotalTrainWordCount

        probabilityWithLinearInterpolation = probabilityWithLinearInterpolation + (param["lambda2"] * (numerator / denominator))

        return math.log(probabilityWithLinearInterpolation, 2)

    def GetEmissionProbability(self, word ,tag, param):
        if(word in self.LowFrequencyWords):
            wordClass = "unk" #self.TranslateToWordClass(word)
            numerator = self.EmissionCount.get((tag,wordClass),0)
        else:
            numerator = self.EmissionCount.get((tag,word),0)

        denominator = self.TagCount[tag]

        if(numerator == 0):
            numerator = numerator + param["smoothingfactor"]
            denominator = denominator + (param["smoothingfactor"]*len(self.TagCount))

        return math.log(numerator / denominator, 2)

    def FindBestTagSequences(self,xDev,param):
        for tokens in xDev:
            tag = self.FindBestTagSequence(tokens,param)
            print(tag)

    def FindBestTagSequence(self,tokenizedSentence,param):

        wordCount = len(tokenizedSentence)
        tagCount = len(self.AllTags)
        scores = []
        backTrack = []

        scores = [[float("-infinity") for x in range(wordCount)] for y in range(tagCount)]
        backTrack = [[0 for x in range(wordCount)] for y in range(tagCount)]

        for tagIndex in range(tagCount):
            scores[tagIndex][0] = 0


        # Start from 1 because the first row belongs to start
        for word in range(1, wordCount):
            for currentTag in range(tagCount):
                for previousTag in range(tagCount):
                    score = scores[previousTag][word-1] + self.GetTransitionProbability(self.AllTags[previousTag], self.AllTags[currentTag], param) + \
                        self.GetEmissionProbability(tokenizedSentence[word][0], self.AllTags[currentTag], param)
                    if score > scores[currentTag][word]:
                      scores[currentTag][word] = score
                      backTrack[currentTag][word] = previousTag

        bestTagIndex = 0
        for tag in range(1, tagCount):
            if scores[tag][wordCount-1] > scores[tag-1][wordCount-1]:
                bestTagIndex=tag

        tags = []
        tagIndex = bestTagIndex
        for t in range(wordCount-1, -1 , -1):
            tags.append(self.AllTags[tagIndex])
            tagIndex = backTrack[tagIndex][t]

        tags.reverse()

        return tags
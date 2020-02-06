from collections import defaultdict
import math

class HmmTrigram(object):
    """A Hmm Trigram language model"""
    
    def __init__(self, xTrain , param):
        self.xTrain = xTrain
        self.TagCount = dict()
        self.TransitionCount = dict()
        self.EmissionCount = dict()
        self.TotalTrainWordCount = 0
        self.Vocabulary = dict()
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
                  self.Vocabulary[word] = self.Vocabulary.get(word,0) + 1
        print("Vocabular count:",len(self.Vocabulary))
        
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
        containsEmoji = False

        for char in word:
            if char.isalpha():
                containsAlpha = True
            if char == ",":
                containsComma = True
            if char == "/":
                containsSlash = True
            if char == "-":
                containsDash = True
            if char == ".":
                containsPeriod = True 
            if( char == "\\u"):
                containsEmoji = True   
 
        if word.startswith("@"): 
            return 'twitterHandle'
        if word.startswith("#"):
            return 'hashTag'
        if word.startswith("\\u"):
            return 'emoticon'
        if containsEmoji:
            return 'containsemoticon'
        if word.startswith("https"):
            return 'uri'
        if word.islower() and word.isalnum() == True: #All lower case
            return 'lowercase'
        if word.isupper() and word.isalnum() == True: #All upper case
            return 'allCaps'
        if word[0].isupper(): 
            return 'initCap'
        if word.isdigit() and len(word) == 2:
            return 'twoDigitNum'
        if word.isdigit() and len(word) == 4:
            return 'fourDigitNum'
        if(word.isalnum() and containsDigits == True):
            return "containsDigitAndAlpha"
        if word.isdigit():
            return 'othernum'
        if(containsDigits ==True and containsSlash == True):
            return "containsDigitAndSlash";
        if(containsDigits ==True and containsDash == True):
            return "containsDigitAndDash";
        if(containsDigits ==True and containsComma == True):
            return "containsDigitAndDash";
        if(containsDigits ==True and containsPeriod == True):
            return "containsDigitAndPeriod";
        else:
            return "other"

    def CalculateCounts(self, xTrain, param):
        for tokens in xTrain:
            for i in range(len(tokens)):
                if(self.Vocabulary.get(tokens[i][0],0) == 0):
                    tokens[i][0] = self.TranslateToWordClass(tokens[i][0])
                self.TotalTrainWordCount = self.TotalTrainWordCount + 1
                self.TagCount[tokens[i][1]] = self.TagCount.get(tokens[i][1],0) + 1
                self.EmissionCount[(tokens[i][1],tokens[i][0])] = self.EmissionCount.get((tokens[i][1],tokens[i][0]),0) + 1
                if(i > 0):
                    self.TransitionCount[(tokens[i - 1][1],tokens[i][1])] = self.TransitionCount.get((tokens[i - 1][1],tokens[i][1]),0) + 1
                if(i > 1):
                    self.TransitionCount[(tokens[i - 2][1],tokens[i - 1][1],tokens[i][1])] = self.TransitionCount.get((tokens[i - 2][1],tokens[i - 1][1],tokens[i][1]),0) + 1

    def GetTransitionProbability(self, conditionState2 , conditionState1, currentState , param):
        probabilityWithLinearInterpolation = 0

        #Trigram part
        numerator = self.TransitionCount.get((conditionState2, conditionState1, currentState),0) 
        denominator = self.TransitionCount.get((conditionState2, conditionState1),0) 
        if(denominator != 0):
            probabilityWithLinearInterpolation = probabilityWithLinearInterpolation + (param["lambda1"] * (numerator / denominator))

        #bigram part
        numerator = self.TransitionCount.get((conditionState1, currentState),0) 
        denominator = self.TagCount.get(conditionState1,0)
        if(denominator != 0):
            probabilityWithLinearInterpolation = probabilityWithLinearInterpolation + (param["lambda2"] * (numerator / denominator))

        #unigram part
        numerator = self.TagCount.get(currentState) 
        denominator = self.TotalTrainWordCount
        if(denominator != 0):
            probabilityWithLinearInterpolation = probabilityWithLinearInterpolation + (param["lambda3"] * (numerator / denominator))

        return math.log(probabilityWithLinearInterpolation, 2)

    def GetEmissionProbability(self, word ,tag, param):
        if(self.Vocabulary.get(word,0) == 0):
            wordClass = self.TranslateToWordClass(word)
            numerator = self.EmissionCount.get((tag,wordClass),0)
        else:
            numerator = self.EmissionCount.get((tag,word),0)

        denominator = self.TagCount[tag]

        if(numerator == 0):
            numerator = numerator + param["smoothingfactor"]
            denominator = denominator + (param["smoothingfactor"]*len(self.TagCount))

        return math.log(numerator / denominator, 2)

    def Tags(self, k):
        if k in (-1, 0):
            return {"<st>"}
        else:
            return self.AllTags

    def FindBestTagSequences(self,xDev,param):
        predictedTags = []
        for tokens in xDev:
            tag = self.FindBestTagSequence(tokens,param)
            predictedTags.append(tag)

        return predictedTags

    def FindBestTagSequence(self,tokenizedSentence,param):
        LOG_PROB_OF_ZERO = -1000
        wordCount = len(tokenizedSentence)
        tagCount = len(self.AllTags)

        pi = defaultdict(float)
        bp = {}

          # Initialization
        pi[(0, "<st>", "<st>")] = 0


        for k in range(1,wordCount+1):
            for u in self.Tags(k-1):
                for v in self.Tags(k):
                    max_score = float('-Inf')
                    max_tag = None
                    for w in self.Tags(k - 2):
                        score = pi[(k-1, w, u)] + self.GetTransitionProbability(w, u,v, param) + \
                            self.GetEmissionProbability(tokenizedSentence[k-1][0], v, param)
                        if score > max_score:
                            max_score = score
                            max_tag = w
                    pi[(k, u, v)] = max_score
                    bp[(k, u, v)] = max_tag

        max_score = float('-Inf')
        u_max, v_max = None, None
        for u in self.Tags(wordCount-1):
            for v in self.Tags(wordCount):
                score = pi[(wordCount, u, v)] + \
                        self.GetTransitionProbability(u, v, "</st>", param) 
                if score > max_score:
                    max_score = score
                    u_max = u
                    v_max = v

        tags = []
        tags.append(v_max)
        tags.append(u_max)

        for i, k in enumerate(range(wordCount-2, 0, -1)):
            tags.append(bp[(k+2, tags[i+1], tags[i])])
        tags.reverse()


        return tags

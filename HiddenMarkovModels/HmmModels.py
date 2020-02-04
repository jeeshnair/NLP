class HmmBigram(object):
    """A Hmm bigram language model"""
    
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

    def FindBestTagSequences(self,xDev,param):
        predictedTags = []
        for tokens in xDev:
            tag = self.FindBestTagSequence(tokens,param)
            predictedTags.append(tag)

        return predictedTags

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

        tagsequence = []
        tagIndex = bestTagIndex
        for t in range(wordCount-1, -1 , -1):
            wordTagPair = []
            wordTagPair.append(tokenizedSentence[t][0])
            wordTagPair.append(self.AllTags[tagIndex])
            wordTagPair.append(tokenizedSentence[t][1])
            tagsequence.append(wordTagPair)
            tagIndex = backTrack[tagIndex][t]

        tagsequence.reverse()

        return tagsequence

class HmmBigram(object):
    """A Hmm bigram language model"""
    
    def __init__(self, xTrain , param):
        self.xTrain = xTrain
        self.TagCount = dict()
        self.TransitionCount = dict()
        self.EmissionCount = dict()
        self.CalculateCounts(xTrain,param)
        self.AllTags = self.TagCount.keys()
        self.TotalTrainWordCount = 0
        self.Vocabulary = []
        self.LowFrequencyWords = []
        self.WordFrequency = dict()
        pass

    def InitializeVocabulary(self,XTrain,param):
        self.CountWordsTrain(xTrain, param)
        for word in self.WordFrequency:
            if self.WordFrequency[word] <= param["oovfrequency"]:
                if(word not in self.LowFrequencyWords):
                    self.LowFrequencyWords.append(word)
            else:
                 self.Vocabulary.append(word)
        
    def CountWordsTrain(self, xTrain,param):
        for tokens in xTrain:
           for i in range(len(tokens)):
                self.WordFrequency[tokens] = self.wordFrequency.get(word, 0) + 1

    def TranslateToWordClass(self,word):
        numDigits = 0
        word = ""
        containsSlash = False
        containsAlpha = False
        containsComma = False
        containsPeriod = False
        containsDash = False
        containsDigits = False

        for char in word:
            if char.isdigit():
                numDigits += 1
                containsDigits = True
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

        digitFraction = numDigits / float(len(word))
        
        if(word.isalnum() and containsDigits == True):
            return "containsDigitAndAlpha"
        if word.isdigit() and len(word) == 2:
            return 'twoDigitNum'
        if word.isdigit() and len(word) == 4:
            return 'fourDigitNum'
        if word.isdigit(): #Is a digit
            return 'othernum'
        if(containsDigits ==True and containsSlash == True and containsAlpha == False):
            return "containsDigitAndSlash";
        if(containsDigits ==True and containsDash == True and containsAlpha == False):
            return "containsDigitAndDash";
        if(containsDigits ==True and containsComma == True and containsAlpha == False):
            return "containsDigitAndDash";
        if(containsDigits ==True and containsPeriod == True and containsAlpha == False):
            return "containsDigitAndPeriod";
        elif digitFraction > 0.5:
            return 'mostlyDigits'
        elif word.islower(): #All lower case
            return 'lowercase'
        elif word.isupper(): #All upper case
            return 'allCaps'
        elif word[0].isupper(): #is a title, initial char upper, then all lower
            return 'initCap'
        elif numDigits > 0:
            return 'containsDigits'  
        else:
            return "other"

    def CalculateCounts(self, xTrain, param):
        for tokens in xTrain:
            for i in range(len(tokens)):
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
        denominator = self.totalTrainWordCount

        probabilityWithLinearInterpolation = probabilityWithLinearInterpolation + (param["lambda2"] * (numerator / denominator))

        return math.log(probabilityWithLinearInterpolation, 2)

    def GetEmissionProbability(self, word ,tag, param):
        if(word not in self.vocabulary):
            wordClass = self.TranslateToWordClass(word)
            numerator = self.EmissionCount[(tag,wordClass)]
        else:
            numerator = self.EmissionCount[(tag,word)]

        denominator = self.TagCount[tag]
        return math.log(numerator / denominator, 2)

    def FindBestTagSequence(self,sentence,param):
        
        words = sentence.split()
        wordCount = len(words)
        tagCount = len(self.AllTags)
        scores = []
        backTrack = []

        scores = [[0.0 for x in range(wordCount)] for y in range(tagCount)]
        backTrack = [[0 for x in range(wordCount)] for y in range(tagCount)]

        # Start from 1 because the first row belongs to start
        for word in range(1, wordCount):
            for currentTag in range(tagCount):
                for previousTag in range(tagCount):
                    score = scores[previousTag][word-1] + self.GetTransitionProbability(self.AllTags[previousTag], self.AllTags[currentTag], param) + self.GetEmissionProbability(elf.AllTags[currentTag], words[i])
                    if score > scores[currentTag][word]:
                      scores[currentTag][word] = score
                      backTrack[currentTag][word] = previousTag

        bestTagIndex = 0
        for tag in range(1, tagCount):
            if scoreMatrix[wordCount-1][tag] > scoreMatrix[wordCount-1][tag-1]:
                bestTagIndex=tag

        tags = []
        tagIndex = bestTagIndex
        for t in range(wordCount,0 , -1):
            tags.append(self.AllTags[tagIndex])
            tagIndex = backpointer[tagIndex][t]
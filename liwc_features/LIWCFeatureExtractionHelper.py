import liwcExtractor as le

class liwcExtractionHelper:
    """A Naive Bayes model for text classification."""

    def __init__(self, liwcPath=None):

        self.liwc = le.liwcExtractor(liwcPath=liwcPath)

    def getTokens(self, text):
        tokens = self.liwc.nltk_tokenize(text)
        return tokens

    def getTokenCount(self, text):
        return len(self.getTokens(text))

    def getLIWCRawFeatures(self, text):
        features = self.liwc.extractFromDoc(text)
        return features

    def getLIWCNormalizedFeatures(self, text):
        features = self.getLIWCRawFeatures(text)
        tokenCount = self.getTokenCount(text)
        normalized_features = [x / float(tokenCount) for x in features]
        #self.liwc.getCategoryIndeces()
        return normalized_features


from gensim import models, similarities
from gensim.corpora import Dictionary
from sklearn.base import BaseEstimator, TransformerMixin


class DocCorpus:
    '''
    This is just the iterator from the tutorial with a couple modifications for cleanliness:
    http://radimrehurek.com/gensim/tut1.html#corpus-streaming-one-document-at-a-time
    '''
    def __init__(self, texts, dict):
        self.texts = texts
        self.dict = dict

    def __iter__(self):
        for line in self.texts:
            yield self.dict.doc2bow(line.lower().split())


class Similarity(BaseEstimator, TransformerMixin):
    def __init__(self, min_token_frequency: int = None):
        """Similarity constructor
        :param min_token_frequency: Minimum number of times that a token must appear to be included
        """
        self.model = models.TfidfModel
        self.similarity = similarities.Similarity
        self.min_token_frequency = min_token_frequency

    def fit(self, tokens):
        return self

    def transform(self, documents):
        dictionary = Dictionary(document.lower().split() for document in documents)

        dictionary = self.filter_tokens(dictionary)
        dictionary.compactify()

        doc_courpus = DocCorpus(documents, dictionary)
        tfidf = self.model(doc_courpus)

        index = self.similarity(corpus=tfidf[doc_courpus], num_features=tfidf.num_nnz, output_prefix='shard')

        return index[tfidf[doc_courpus]]

    def filter_tokens(self, dictionary: Dictionary) -> Dictionary:
        """Filter tokens according to some conditions
        :param dictionary: Token dictionary
        :return: Filtered token dictionary
        """
        if self.min_token_frequency and self.min_token_frequency > 1:
            once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < self.min_token_frequency]
            dictionary.filter_tokens(once_ids)

        return dictionary

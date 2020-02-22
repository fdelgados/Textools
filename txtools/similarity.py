from gensim import models, similarities
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
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
            yield self.dict.doc2bow(line)


class NearDuplicate(BaseEstimator, TransformerMixin):
    DEFAULT_PHRASE_MIN_COUNT = 1
    DEFAULT_PHRASE_THRESHOLD = 10.0
    MATRIX_SIMILARITY = 'matrix'
    SPARSE_MATRIX_SIMILARITY = 'sparse'

    SIMILARITY_MATRIX_TYPES = [
        MATRIX_SIMILARITY,
        SPARSE_MATRIX_SIMILARITY,
    ]

    def __init__(self, phrase_min_count: int = None, phrase_threshold: float = None, similarity_class: str = None):
        """NearDuplicate constructor
        :param phrase_min_count: Ignore all words and bigrams with total collected count lower than this value.
        :param phrase_threshold : Represent a score threshold for forming the phrases (higher means fewer phrases).
            A phrase of words `a` followed by `b` is accepted if the score of the phrase is greater than threshold.
        """
        self.dictionary = None
        self.texts_bigrams = []
        self.similarity_class = similarity_class
        self.similarity = self.__create_similarity_matrix__()
        self.phrase_min_count = phrase_min_count if phrase_min_count else self.DEFAULT_PHRASE_MIN_COUNT
        self.phrase_threshold = phrase_threshold if phrase_threshold else self.DEFAULT_PHRASE_THRESHOLD

    def __create_similarity_matrix__(self):
        """Creates a similarity matrix
        :return: Similarity Matrix
        """
        if self.similarity_class == self.MATRIX_SIMILARITY:
            return similarities.MatrixSimilarity

        if self.similarity_class == self.SPARSE_MATRIX_SIMILARITY:
            return similarities.SparseMatrixSimilarity

        return similarities.Similarity

    def fit(self, documents, labels=None):
        return self

    def transform(self, documents):
        texts = [[text for text in simple_preprocess(document, deacc=True)]
                 for document in documents]

        bigram = Phrases(texts, min_count=self.phrase_min_count, threshold=self.phrase_threshold)

        bigram_phraser = Phraser(bigram)
        self.texts_bigrams = [[text for text in bigram_phraser[simple_preprocess(doc, deacc=True)]]
                              for doc in documents]

        self.dictionary = Dictionary(self.texts_bigrams)
        self.dictionary.compactify()

        corpus = [self.dictionary.doc2bow(doc_string) for doc_string in self.texts_bigrams]

        index = self.__build_index__(corpus)

        return index[corpus]

    def __build_index__(self, corpus):
        """Builds the index depending on similarity matrix type
        :param corpus: Document corpus
        :return: Similarities index
        """
        if self.similarity_class in self.SIMILARITY_MATRIX_TYPES:
            return self.similarity(corpus=corpus, num_features=len(self.dictionary))

        return self.similarity(corpus=corpus, num_features=len(self.dictionary), output_prefix='shard')


class Similarity(BaseEstimator, TransformerMixin):

    def __init__(self, min_token_frequency: int = None):
        """Similarity constructor
        :param min_token_frequency: Minimum number of times that a token must appear to be included
        """
        self.dictionary = None
        self.model = models.TfidfModel
        self.similarity = similarities.Similarity
        self.min_token_frequency = min_token_frequency

    def fit(self, documents, labels=None):
        self.dictionary = Dictionary(document.lower().split() for document in documents)

        self.filter_tokens()
        self.dictionary.compactify()

        return self

    def transform(self, documents):
        doc_courpus = DocCorpus(documents, self.dictionary)
        tfidf = self.model(doc_courpus)

        index = self.similarity(corpus=tfidf[doc_courpus], num_features=tfidf.num_nnz, output_prefix='shard')

        return index[tfidf[doc_courpus]]

    def filter_tokens(self):
        if self.min_token_frequency and self.min_token_frequency > 1:
            once_ids = [tokenid for tokenid, docfreq in self.dictionary.dfs.items()
                        if docfreq < self.min_token_frequency]
            self.dictionary.filter_tokens(once_ids)

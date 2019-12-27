import numpy as np
from gensim import corpora, models, similarities
from sklearn.base import BaseEstimator, TransformerMixin
from .normalizer import TextNormalizer


class Similarity(BaseEstimator, TransformerMixin):
    DEFAULT_LANGUAGE = 'en'

    def __init__(self, lang: str = None):
        if not lang:
            lang = self.DEFAULT_LANGUAGE

        self.lang = lang
        self.model = models.TfidfModel
        self.similarity_matrix = similarities.MatrixSimilarity

    def fit(self, tokens):
        return self

    def transform(self, documents):
        sims = np.empty((0, len(documents)), dtype=float)

        if not self.documents_are_tokenized(documents):
            documents = self.normalize_documents(documents)

        dictionary = corpora.Dictionary(documents)
        num_features = len(dictionary.token2id)
        corpus = [dictionary.doc2bow(document) for document in documents]

        tfidf = self.model(corpus)
        index = self.similarity_matrix(tfidf[corpus], num_features=num_features)

        for idx, document in enumerate(documents):
            document_bow = dictionary.doc2bow(document)
            sim = index[tfidf[document_bow]]
            sims = np.append(sims, [sim], axis=0)

        return sims

    def normalize_documents(self, documents):
        text_normalizer = TextNormalizer(self.lang)

        return [text_normalizer.normalize(document) for document in documents]

    def documents_are_tokenized(self, documents):
        return all(isinstance(document, np.ndarray) for document in documents)

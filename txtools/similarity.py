import numpy as np
from gensim import corpora, models, similarities
from sklearn.base import BaseEstimator, TransformerMixin


class Similarity(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = models.TfidfModel
        self.similarity_matrix = similarities.MatrixSimilarity

    def fit(self, documents):
        return self

    def transform(self, documents):
        sims = np.empty((0, len(documents)), dtype=float)

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

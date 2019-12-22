import nltk
from nltk.cluster import KMeansClusterer
from sklearn.base import BaseEstimator, TransformerMixin


class KMeansClusters(BaseEstimator, TransformerMixin):
    def __init__(self, k: int = 7):
        self.k = k
        self.distance = nltk.cluster.util.cosine_distance
        self.model = KMeansClusterer(self.k, self.distance, avoid_empty_clusters=True)

    def fit(self, documents):
        return self

    def transform(self, documents):
        return self.model.cluster(documents, assign_clusters=True)

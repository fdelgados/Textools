from gensim import models, similarities
from gensim.corpora import Dictionary
from sklearn.base import BaseEstimator, TransformerMixin
from .normalizer import TextNormalizer, clean_text


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
    DEFAULT_LANGUAGE = 'en'

    def __init__(self, lang: str = None, clean: bool = False, min_token_frequency: int = None):
        if not lang:
            lang = self.DEFAULT_LANGUAGE

        self.lang = lang
        self.model = models.TfidfModel
        self.similarity = similarities.Similarity
        self.clean = clean
        self.min_token_frequency = min_token_frequency
        self.cleaners = ['html_tags', 'html_entities', 'unicode_nbsp', 'non_ascii', 'punctuation']

    def fit(self, tokens):
        return self

    def transform(self, documents):
        if self.clean:
            documents = self.clean_documents(documents)

        dictionary = Dictionary(document.lower().split() for document in documents)

        dictionary = self.filter_tokens(dictionary)
        dictionary.compactify()

        doc_courpus = DocCorpus(documents, dictionary)
        tfidf = self.model(doc_courpus)

        index = self.similarity(corpus=tfidf[doc_courpus], num_features=tfidf.num_nnz, output_prefix='shard')

        return index[tfidf[doc_courpus]]

    def filter_tokens(self, dictionary: Dictionary) -> Dictionary:
        punct_ids = [tokenid for tokenid, token in dictionary.items() if TextNormalizer.is_punct(token)]
        dictionary.filter_tokens(punct_ids)

        if self.min_token_frequency and self.min_token_frequency > 1:
            once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < self.min_token_frequency]
            dictionary.filter_tokens(once_ids)

        return dictionary

    def clean_documents(self, documents):
        return [clean_text(document, cleaners=self.cleaners) for document in documents]

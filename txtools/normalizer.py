import re
import html
import unicodedata
import numpy as np
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


class TextNormalizer(BaseEstimator, TransformerMixin):
    ALLOWED_LANGUAGES = {'en': 'english',
                         'es': 'spanish',
                         'fr': 'french',
                         'de': 'german',
                         'it': 'italian'}
    DEFAULT_LANGUAGE = 'en'

    WORD_PATTERN = r'\W+'
    NON_WORD_PATTERN = r'[\W_]+'

    def __init__(self, language: str = None):
        """
        :param language: Two letter language code. ISO 639-1: https://en.wikipedia.org/wiki/ISO_639-1
        :raises ValueError: Raises an exception if the language code is not valid
        """
        if not language:
            language = self.DEFAULT_LANGUAGE

        self.__guard_against_not_allowed_language__(language)

        self.language = language
        self.stopwords = set(stopwords.words(self.ALLOWED_LANGUAGES[self.language]))
        self.lemmatizer = WordNetLemmatizer()

    def __guard_against_not_allowed_language__(self, language: str):
        """ Raises an exception if the language code is not valid
        :param language: Two letter language code. ISO 639-1: https://en.wikipedia.org/wiki/ISO_639-1
        :raises ValueError
        """
        if language not in self.ALLOWED_LANGUAGES.keys():
            allowed_languages = [allowed_language.capitalize() for allowed_language in self.ALLOWED_LANGUAGES.values()]
            allowed_languages = ', '.join(allowed_languages)

            raise ValueError('{} is not an allowed language. Allowed languages are: {}.'.format(language.capitalize(),
                                                                                                allowed_languages))

    @staticmethod
    def is_punct(token: str) -> bool:
        """ Checks if all chars in token are punctuation symbols
        :param token: Token
        :return: True if all chars in token are punctuation symbols, False otherwise
        """
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self, token: str) -> bool:
        """ Checks if token is a stop word
        :param token: Token
        :return: True if token is a stop word, False otherwise
        """
        return token.lower() in self.stopwords

    def normalize(self, text: str, clean: bool = False) -> np.ndarray:
        """ Normalize text
        :param text: Text to be normalized
        :param clean: Whether or not text has to be cleaned
        :return: Array of normalized tokens
        """

        if clean:
            text = clean_text(text)

        sentences = self.tokenize(text)

        return np.array([
            self.lemmatize(token, tag).lower()
            for sentence in sentences
            for (token, tag) in sentence
            if not TextNormalizer.is_punct(token) and not self.is_stopword(token)
        ])

    def tokenize(self, text: str) -> List[Tuple[str, str]]:
        """ Splits text in a list of tuples composed by token and his part of speech tag
        :param text: Text to be tokenized
        :return: List of tuples composed by token and his part of speech tag
        """
        return [
            nltk.pos_tag(nltk.wordpunct_tokenize(sentence))
            for sentence in nltk.sent_tokenize(text, self.ALLOWED_LANGUAGES[self.language])
        ]

    def lemmatize(self, token, pos_tag):
        """ Lemmatize token
        :param token: Token
        :param pos_tag: Part-of-speech tag
        :return: Lemmatized word
        """
        tag = {
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV,
            'J': wordnet.ADJ
        }.get(pos_tag[0], wordnet.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def fit(self, X, y=None):
        return self

    def transform(self, documents):
        return [self.normalize(doc) for doc in documents]


def clean_text(text: str) -> str:
    """ Removes unwanted chars from text
    :param text: Text to be cleaned
    :return: Clean text
    """
    text = remove_html_tags(text)
    text = decode_html_entities(text)
    text = remove_unicode_nbsp(text)
    text = remove_control_chars(text)
    text = remove_extra_quotation(text)
    text = remove_non_ascii(text)
    text = remove_extra_whitespaces(text)
    text = remove_urls(text)

    return text


def remove_html_tags(text: str) -> str:
    """ Removes html tags
    :param text: Text to be cleaned
    :return: Clean text
    """
    clean = re.compile(r'<.*?>')

    return re.sub(clean, '', text)


def remove_extra_whitespaces(text: str) -> str:
    """ Removes extra whitespaces
    :param text: Text to be cleaned
    :return: Clean text
    """
    return re.sub(r' +', ' ', text)


def remove_extra_quotation(text: str) -> str:
    """ Removes extra quotation marks
    :param text: Text to be cleaned
    :return: Clean text
    """
    text = re.sub(r'\"{2,}', '"', text)

    return re.sub(r'\'{2,}', "'", text)


def remove_control_chars(text: str) -> str:
    """ Removes control chars
    :param text: Text to be cleaned
    :return: Clean text
    """
    return text.translate(str.maketrans('\n\t\r', '   '))


def remove_unicode_nbsp(text: str) -> str:
    """ Removes unicode whitespaces
    :param text: Text to be cleaned
    :return: Clean text
    """
    return text.replace(u'\xa0', u' ')


def decode_html_entities(text: str) -> str:
    """ Converts html entities in the corresponding unicode string
    :param text: Text to be cleaned
    :return: Clean text
    """
    return html.unescape(text)


def remove_non_ascii(text: str) -> str:
    """ Removes non ascii characters
    :param text: Text to be cleaned
    :return: Clean text
    """
    return ''.join(char for char in text if ord(char) < 128)


def remove_urls(text):
    """ Removes all urls from text
    :param text: The string being searched and replaced on
    :return: Text without the urls
    """
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, '')

    return text


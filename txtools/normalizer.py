import sys
import re
import html
import unicodedata
import string
from typing import List, Tuple
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
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
                         'it': 'italian',
                         'ca': 'catalan',
                         'pt': 'portuguese'}
    DEFAULT_LANG_CODE = 'en'

    LEMMATIZATION = 'lemmatization'
    STEMMING = 'stemming'

    DEFAULT_NORMALIZATION_METHOD = 'stemming'

    def __init__(self, method: str = None, lang_code: str = None):
        """TextNormalizer contructor
        :param method: Normalization method: stemming or lemmatization
        :param lang_code: ISO 639-1 code language
        """
        if method != self.LEMMATIZATION and method != self.STEMMING:
            raise ValueError('Invalid reduction method')

        self.language = self.__get_language__(lang_code)
        self.normalization_method = self.__get_normalization_method__(method)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words(self.language)

    def __get_language__(self, lang_code: str) -> str:
        """Return the language name from language code
        :param lang_code: ISO 639-1 code language
        :return: Language name
        :raises: ValueError
        """

        if lang_code not in self.ALLOWED_LANGUAGES.keys():
            raise ValueError('{} is not a supported language code'.format(lang_code))

        return self.ALLOWED_LANGUAGES[lang_code]

    def __get_normalization_method__(self, normalization_method: str = None):
        """Returns the normalization method
        :param normalization_method: Normalization method to check
        :return: Normalization method
        """
        if not normalization_method:
            return self.DEFAULT_NORMALIZATION_METHOD

        if normalization_method != self.LEMMATIZATION and normalization_method != self.STEMMING:
            raise ValueError('Invalid normalization method')

        return normalization_method

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
        return token.lower() in self.stop_words

    def normalize(self, text: str) -> str:
        """ Normalize text
        :param text: Text to be normalized
        :return: Normalized text
        """
        text = clean_text(text)

        if self.normalization_method == self.LEMMATIZATION:
            return self.normalize_with_lemmatization(text)
        elif self.normalization_method == self.STEMMING:
            return self.normalize_with_stemming(text)

    def normalize_with_lemmatization(self, text: str) -> str:
        """Return normalized text by lemmatization method
        :param text: Text to normalize
        :return: Normalized text
        """
        sentences = self.sentence_tokenize(text)

        normalized_tokens = [
            self.lemmatize(token, tag).lower()
            for sentence in sentences
            for (token, tag) in sentence
            if not TextNormalizer.is_punct(token) and not self.is_stopword(token)
        ]

        return ' '.join([' '.join(tokens) for tokens in normalized_tokens])

    def sentence_tokenize(self, text: str) -> List[Tuple[str, str]]:
        """ Splits text in a list of tuples composed by token and his part of speech tag
        :param text: Text to be tokenized
        :return: List of tuples composed by token and his part of speech tag
        """
        return [
            nltk.pos_tag(nltk.wordpunct_tokenize(sentence))
            for sentence in nltk.sent_tokenize(text, language=self.language)
        ]

    def normalize_with_stemming(self, text: str) -> str:
        """Normalize text by stemming method
        :param text: Text to normalize
        :return: Normalized text
        """
        tokenizer = RegexpTokenizer(r'\w+')
        stemmer = SnowballStemmer(self.language)

        tokens = tokenizer.tokenize(text.lower())

        tokens = [token for token in tokens
                  if not self.is_stopword(token) and not TextNormalizer.is_punct(token)]

        tokens_stemmed = [stemmer.stem(token) for token in tokens]

        return ' '.join(tokens_stemmed)

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


def clean_text(text: str, cleaners: List[str] = None, exclude: List[str] = None) -> str:
    """ Removes unwanted chars from text
    :param text: Text to be cleaned
    :param cleaners: List of cleaners to be applied to text
    :param exclude: List of cleaners that wont be applied
    :return: Clean text
    """
    if not cleaners:
        cleaners = ['html_tags', 'html_entities', 'unicode_nbsp', 'tabs', 'new_line'
                    'extra_quotation', 'non_ascii', 'extra_whitespaces', 'urls', 'punctuation']

    if exclude:
        cleaners = [cleaner for cleaner in cleaners if cleaner not in exclude]

    for cleaner in cleaners:
        cleaner_func_name = 'remove_{}'.format(cleaner)
        try:
            cleaner_function = getattr(sys.modules[__name__], cleaner_func_name)
        except AttributeError:
            continue

        text = cleaner_function(text)

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


def remove_new_line(text: str) -> str:
    """ Removes new line chars
    :param text: Text to be cleaned
    :return: Clean text
    """
    return text.translate(str.maketrans('\n\r', '  '))


def remove_tabs(text: str) -> str:
    """ Removes tabs
    :param text: Text to be cleaned
    :return: Clean text
    """
    return text.replace('\t', ' ')


def remove_unicode_nbsp(text: str) -> str:
    """ Removes unicode whitespaces
    :param text: Text to be cleaned
    :return: Clean text
    """
    return text.replace(u'\xa0', u' ')


def remove_html_entities(text: str) -> str:
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


def remove_punctuation(text: str) -> str:
    """ Removes punctuation from text
    :param text: The string being searched and replaced on
    :return: Text without the punctuation characters
    """
    punctuation = string.punctuation + '¿¡'
    table = str.maketrans('', '', punctuation)
    words = text.split()

    stripped = [word.translate(table) for word in words]

    return ' '.join(stripped)


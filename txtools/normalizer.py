import re
import string
import datetime
from typing import List, Union

from .cleaners import decode_html_entities
import unicodedata
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import contractions

nltk.download('punkt')
nltk.download('stopwords')


class TextNormalizer:
    ALLOWED_LANGUAGES = {'en': 'english',
                         'es': 'spanish',
                         'fr': 'french',
                         'de': 'german',
                         'it': 'italian'}
    DEFAULT_LANGUAGE = 'en'

    WORD_PATTERN = r'\W+'
    NON_WORD_PATTERN = r'[\W_]+'

    def __init__(self, language: str = None):
        if not language:
            language = self.DEFAULT_LANGUAGE

        self.__guard_against_not_allowed_language__(language)

        self.language = language

    def __guard_against_not_allowed_language__(self, language):
        if language not in self.ALLOWED_LANGUAGES.keys():
            allowed_languages = [allowed_language.capitalize() for allowed_language in self.ALLOWED_LANGUAGES.values()]
            allowed_languages = ', '.join(allowed_languages)

            raise ValueError('{} is not an allowed language. Allowed languages are: {}.'.format(language.capitalize(),
                                                                                                allowed_languages))

    def normalize(self,
                  corpus: List[str],
                  lemmatize: bool = False,
                  stem: bool = False,
                  remove_digits: bool = True,
                  remove_accents: bool = False,
                  tokenize: bool = False,
                  only_text_chars: bool = False,
                  silent: bool = True) -> List[str]:

        normalized_corpus = []

        total_length = len(corpus)
        counter = 1
        for text in corpus:
            normalized_text = self.normalize_text(text,
                                                  lemmatize,
                                                  stem,
                                                  remove_digits,
                                                  remove_accents,
                                                  tokenize,
                                                  only_text_chars)

            normalized_corpus.append(normalized_text)
            now = datetime.datetime.now()

            if not silent:
                print('[{}] Normalizing {}/{}'.format(now.strftime('%Y-%m-%d %H:%M:%S'), counter, total_length))

            counter += 1

        return normalized_corpus

    def normalize_text(self,
                       text: str,
                       lemmatize: bool = False,
                       stem: bool = False,
                       remove_digits: bool = True,
                       remove_accents: bool = False,
                       tokenize: bool = False,
                       only_text_chars: bool = False) -> Union[str, List[str]]:

        text = decode_html_entities(text)

        if self.language == 'en':
            text = contractions.fix(text)

        text = self.__lexical_normalization__(text,
                                              lemmatize=lemmatize,
                                              stem=stem)
        text = self.remove_special_characters(text,
                                              remove_digits=remove_digits,
                                              remove_accents=remove_accents)

        if only_text_chars:
            text = self.__keep_text_characters__(text)

        text = self.remove_stop_words(text)

        if not tokenize:
            return text

        return self.tokenize_text(text)

    def __lexical_normalization__(self, text: str, lemmatize: bool, stem: bool) -> str:
        if lemmatize:
            return self.lemmatize_text(text)

        if stem:
            return self.stem_text(text)

        return text.lower()

    def __keep_text_characters__(self, text: str) -> str:
        tokens = self.tokenize_text(text)

        filtered_tokens = [token for token in tokens if token.isalpha()]

        return ' '.join(filtered_tokens)

    def stem_text(self, text: str) -> str:
        if self.language == 'en':
            stemmer = PorterStemmer()
        else:
            stemmer = SnowballStemmer(self.ALLOWED_LANGUAGES[self.language])

        try:
            tokens = self.tokenize_text(text)
        except ValueError:
            print('<<{}>>'.format(text))
            return ''

        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        return ' '.join(stemmed_tokens)

    def lemmatize_text(self, text: str) -> str:
        nlp = spacy.load(self.language)
        tokens = nlp(text)

        lemmatized_tokens = [token.lemma_ for token in tokens]
        lemmatized_text = ' '.join(lemmatized_tokens)

        return lemmatized_text

    def remove_special_characters(self, text: str, remove_digits: bool = True, remove_accents: bool = False) -> str:
        try:
            tokens = self.tokenize_text(text)
        except ValueError:
            print('<<{}>>'.format(text))
            return ''

        punctuation = '{}{}'.format(string.punctuation, '¡¿ºª')

        filtered_tokens = [token.translate(str.maketrans('', '', punctuation)) for token in tokens]

        if remove_digits:
            filtered_tokens = [token for token in filtered_tokens if re.match(r'^\d+$', token) is None]

        text = ' '.join(filtered_tokens)

        if not remove_accents:
            return text

        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def remove_digits(self, text: str) -> str:
        try:
            tokens = self.tokenize_text(text)
        except ValueError:
            print('<<{}>>'.format(text))
            return ''

        filtered_tokens = [token for token in tokens if re.match(r'^\d+$', token) is None]

        return ' '.join(filtered_tokens)

    def remove_stop_words(self, text: str) -> str:
        try:
            tokens = self.tokenize_text(text)
        except ValueError:
            print('<<{}>>'.format(text))
            return ''

        stop_words = stopwords.words(self.ALLOWED_LANGUAGES[self.language])

        filtered_tokens = [token for token in tokens if token not in stop_words]

        return ' '.join(filtered_tokens)

    def tokenize_text(self, text: str) -> List[str]:
        if not text:
            raise ValueError('The text to be preprocessed can not be empty.')

        return word_tokenize(text, language=self.ALLOWED_LANGUAGES[self.language])

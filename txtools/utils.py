import spacy
from spacy_langdetect import LanguageDetector


class LangDetector:
    MIN_CONFIDENCE = 70
    DEFAULT_LANGUAGE = 'en'

    def __init__(self, language: str = None):
        self.language = language if language is not None else self.DEFAULT_LANGUAGE

    def lang(self, text: str, min_confidence: int = None) -> str:
        if not min_confidence:
            min_confidence = self.MIN_CONFIDENCE

        nlp = spacy.blank(self.DEFAULT_LANGUAGE)
        
        nlp.add_pipe(nlp.create_pipe('sentencizer'))
        nlp.add_pipe(LanguageDetector(),
                     name='language_detector',
                     last=True)
        doc = nlp(text)

        detected_language = doc._.language.get('language')
        detection_confidence = doc._.language.get('score') * 100

        return detected_language if detection_confidence >= min_confidence else self.language

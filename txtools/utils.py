import spacy
from spacy_langdetect import LanguageDetector


class LangDetector:
    MIN_CONFIDENCE = 70
    DEFAULT_LANGUAGE = 'en'
    UNKNOWN = 'xx'

    SUPPORTED_LANGUAGES = {'af': 'Afrikaans', 'ar': 'Arabic', 'bg': 'Bulgarian', 'bn': 'Bengali', 'ca': 'Catalan',
                           'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German', 'el': 'Greek', 'en': 'English',
                           'es': 'Spanish', 'et': 'Estonian', 'fa': 'Persian', 'fi': 'Finnish', 'fr': 'French',
                           'gu': 'Gujarati', 'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian',
                           'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese', 'kn': 'Kannada', 'ko': 'Korean',
                           'lt': 'Lithuanian', 'lv': 'Latvian', 'mk': 'Macedonian', 'ml': 'Malayalam', 'mr': 'Marathi',
                           'ne': 'Nepali', 'nl': 'Dutch', 'no': 'Norwegian', 'pa': 'Punjabi', 'pl': 'Polish',
                           'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sk': 'Slovak', 'sl': 'Slovenian',
                           'so': 'Somali', 'sq': 'Albanian', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil',
                           'te': 'Telugu', 'th': 'Thai', 'tl': 'Tagalog', 'tr': 'Turkish', 'uk': 'Ukrainian',
                           'ur': 'Urdu', 'vi': 'Vietnamese', 'zh-cn': 'Chinese (Simplified)',
                           'zh-tw': 'Chinese (Traditional)', 'xx': 'Unknown'}

    def iso_639_1_code(self, text: str, min_confidence: int = None) -> str:
        """ Returns ISO 639-1 code language of text
        :param text: Text we want to translate
        :param min_confidence: Minimum percentage of confidence from which we consider the detection to be correct
        :return: Detected language in two-letter code format (ISO 639-1) or 'xx' if the detection does not reach the
        minimum confidence
        """
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

        return detected_language if detection_confidence >= min_confidence else self.UNKNOWN

    def lang(self, text: str, min_confidence: int = None) -> str:
        """ Returns the language name of text
        :param text: Text we want to translate
        :param min_confidence: Minimum percentage of confidence from which we consider the detection to be correct
        :return: Detected language name or 'Unknown' if the detection does not reach the minimum confidence
        """
        return self.SUPPORTED_LANGUAGES[self.iso_639_1_code(text, min_confidence)]

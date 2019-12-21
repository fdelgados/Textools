import re
import html


def strip_html_tags(text):
    clean = re.compile(r'<.*?>')

    return re.sub(clean, '', text)


def strip_extra_whitespaces(text):
    return re.sub(r' +', ' ', text)


def strip_extra_quotation(text):
    text = re.sub(r'\"{2,}', '"', text)

    return re.sub(r'\'{2,}', "'", text)


def strip_control_chars(text):
    return text.translate(str.maketrans('\n\t\r', '   '))


def remove_unicode_nbsp(text):
    text = text.replace(u'\xa0', u' ')

    return text.encode('utf-8')


def decode_html_entities(text):
    return html.unescape(text)

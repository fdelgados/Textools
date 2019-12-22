# Text Clean and Preprocess Tools (texcptulz)

textcptulz is a text preprocessing framework to transform raw ingested text into a form that is ready for computation and modeling.

## Table of Contents
* [Installations](#installations)
* [Project Motivation](#project-motivation)
* [Modules](#modules)
* [Instructions](#instructions)
* [License](#license)

## Installations
### Dependencies

textcptulz requires:

* Python (>=3.5)
* NumPy
* scikit-learn
* NLTK
* spaCy
* spacy-langdetect
* gensim

### Installation

You can install texcptulz using `pip`
```
pip install texcptulz
```
That easy.

## Project Motivation

texcptulz aims to be a wrapper for all that processes involved in the wrangling part of an ETL pipeline for text analysis.
In addition, it includes other useful tools such as the detection of the language of a document. The languages supported are the following:

* English
* Spanish
* French
* German
* Italian

## Modules

The modules included in texcptulz are:

### `normalizer`
The `normalizer` module includes `TextNormalizer` class that can perform the text normalization process, taking a text
as input and returning a list of lemmatized tokens
```python
from txtools.normalizer import TextNormalizer

text = 'Python is a programming language that lets you work quickly' \ 
' and integrate systems more effectively'

normalizer = TextNormalizer()
tokens = normalizer.normalize(text, clean=True)

print(tokens)

# output
# ['python', 'programming', 'language', 'let', 'work', 'quickly', 'integrate', 'system', 'effectively']
```
* clusterer:
* similarity:
* vectorizer:
* utils:

## Instructions
## License

Copyright (c) 2019 Cisco Delgado

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

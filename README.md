# Text Clean and Preprocess Tools (texcptulz)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/fdelgados/Textools?style=flat)

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
In addition, it includes other useful tools such as the detection of the language of a document. These are some of the 
supported languages:

* English
* Spanish
* French
* German
* Italian

## Modules

The modules included in texcptulz are:

### `normalizer`
The `normalizer` module includes `TextNormalizer` class that can perform the text normalization process, taking a text
as input and returning a list of lemmatized tokens.

Special characters, HTML tags, urls, etc. will be removed, using the `clean_text` function, then, the text will be 
converted to lowercase and will be splitted into tokens and lemmatized. 

### `clusterer`
The `clusterer` module has only one class `KMeansClusters` that performs k-means clustering. The distance measure used 
is cosine distance. The `k` parameter is the number of clusters, the default value is 7.

### `similarity`
The `similarity` module has only one class, `Similarity` that creates a `mxm` similarity matrix for a group of documents. 
In this matrix, columns and rows represents the index of each document the elements of the matrix are similarity between 
documents.

The range of values will be between 0 and 1, where 0 means totally different and 1 means that the content is identical.

### `vectorizer`
The `vectorizer` module has only one class, `OneHotVectorizer` that performs the text conversion into a vector 
representations of categorical variables as binary vectors (one hot encoding)

### `utils`
This module includes the `LangDetector` class to detect the language of a document

## Instructions
### Normalize text

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
`TextNormalizer` class implements the `Transformer` interface, this allows us to add this class into a scikit-learn pipeline.

### Language detection
`utils` module includes the `LangDetector` class intended to detect the language in which a document was written. The 
detection is done with a certain confidence. You can provide a minimum percentage of confidence for a detection process:

```python
from txtools.utils import LangDetector

documents = [
    'El aprendizaje automático tiene una amplia gama de aplicaciones, incluyendo motores de búsqueda, diagnósticos médicos, detección de fraude en el uso de tarjetas de crédito',
    'Machine learning tasks are classified into several broad categories. In supervised learning, the algorithm builds a mathematical model from a set of data',
    'L\'apprendimento automatico viene impiegato in quei campi dell\'informatica nei quali progettare e programmare algoritmi espliciti è impraticabile',
    'Selon les informations disponibles durant la phase d\'apprentissage, l\'apprentissage est qualifié de différentes manières.'
]

lang_detector = LangDetector()

for idx, document in enumerate(documents):
    print('Document {} is written in {}'.format(idx, lang_detector.lang(document)))

# output
# Document 0 is written in Spanish
# Document 1 is written in English
# Document 2 is written in Italian
# Document 3 is written in French
```
Also, you can get the ISO 639-1 code:

```python

for idx, document in enumerate(documents):
    print('ISO 639-1 code for document {} is {}'.format(idx, lang_detector.iso_639_1_code(document)))

# output
# ISO 639-1 lang code for document 0 is es
# ISO 639-1 lang code for document 1 is en
# ISO 639-1 lang code for document 2 is it
# ISO 639-1 lang code for document 3 is fr
```

### Clustering documents

```python
from txtools.clusterer import KMeansClusters

documents = [] # this must contains a lot of documents

k_means = KMeansClusters(k=10)

clusters = k_means.transform(documents)
``` 
### Compute document similarity
`Similarity` creates creates a `mxm` similarity matrix for a corpus where `m` is the number of documents.

```python
from txtools.similarity import Similarity

documents = [
    'Psycho is a 1960 American psychological horror film directed and produced by Alfred Hitchcock, and written by Joseph Stefano.',
    'North by Northwest is a 1959 American thriller film directed by Alfred Hitchcock, starring Cary Grant, Eva Marie Saint and James Mason.',
    'The Birds is a 1963 American horror-thriller film directed and produced by Alfred Hitchcock.',
    'Rear Window is a 1954 American Technicolor mystery thriller film directed by Alfred Hitchcock and written by John Michael Hayes based on Cornell Woolrich\'s 1942 short story "It Had to Be Murder".'
]

similarity = Similarity()
sims = similarity.transform(documents)

print(sims)

# output

# [[0.99999994 0.         0.13075474 0.02665992]
#  [0.         1.         0.00812627 0.00331377]
#  [0.13075474 0.00812627 1.         0.0069054 ]
#  [0.02665992 0.00331377 0.0069054  1.        ]]
```

### Using a scikit-learn pipeline
`TextNormalizer`, `OneHotVectorizer`, `KMeansClusters` and `Similarity` implement the `Transformer` interface, so we can
add them to a scikit-learn pipeline.

```python
from sklearn.pipeline import Pipeline
from txtools.normalizer import TextNormalizer
from txtools.similarity import Similarity

documents = [
    'Psycho is a 1960 American psychological horror film directed and produced by Alfred Hitchcock, and written by Joseph Stefano.',
    'North by Northwest is a 1959 American thriller film directed by Alfred Hitchcock, starring Cary Grant, Eva Marie Saint and James Mason.',
    'The Birds is a 1963 American horror-thriller film directed and produced by Alfred Hitchcock.',
    'Rear Window is a 1954 American Technicolor mystery thriller film directed by Alfred Hitchcock and written by John Michael Hayes based on Cornell Woolrich\'s 1942 short story "It Had to Be Murder".'
]

model = Pipeline([
    ('norm', TextNormalizer()),
    ('sim', Similarity())
])

sims = model.fit_transform(documents)

```

### Cleaning text
You can clean text with the `clean_text` function included in the `normalizer` module

```python
from txtools.normalizer import clean_text

text = '<p><i><b>2001: A    Space Odyssey</b></i> \n\nis a 1968 <a href="/wiki/Epic_film" title="Epic ' \
           'film">epic</a> <a href="/wiki/Science_fiction_film" title="Science fiction film">science fiction film</a> ' \
           'produced and directed by\t <a href="/wiki/Stanley_Kubrick" title="Stanley Kubrick">Stanley Kubrick</a>. ' \
           'The screenplay was written by Kubrick and <a href="/wiki/Arthur_C._Clarke" title="Arthur C. ' \
           'Clarke">Arthur C. Clarke</a>, and was inspired by Clarke\'s short story ""<a href="/wiki/The_Sentinel_(' \
           'short_story)" title="The Sentinel (short story)">The Sentinel</a>" and other short stories by Clarke. A ' \
           '<a href="/wiki/2001:_A_Space_Odyssey_(novel)" title="2001: A Space Odyssey (novel)">novelisation of the ' \
           'film</a> released after the film\'s premiere was in part written concurrently with the screenplay. The ' \
           'film, which follows a voyage to <a href="/wiki/Jupiter" title="Jupiter">Jupiter</a> with the <a ' \
           'href="/wiki/Sentience" title="Sentience">sentient</a> computer <a href="/wiki/HAL_9000" title="HAL ' \
           '9000">HAL</a> after the discovery of a <a href="/wiki/Monolith_(Space_Odyssey)" title="Monolith (Space ' \
           'Odyssey)">featureless alien monolith</a> affecting human evolution, deals with themes of <a ' \
           'href="/wiki/Existentialism" title="Existentialism">existentialism</a>, <a href="/wiki/Human_evolution" ' \
           'title="Human evolution">human evolution</a>, technology, <a href="/wiki/Artificial_intelligence" ' \
           'title="Artificial intelligence">artificial intelligence</a>, and the possibility of <a ' \
           'href="/wiki/Extraterrestrial_life" title="Extraterrestrial life">extraterrestrial life</a>.</p> '

print(clean_text(text))

# output

# 2001: A Space Odyssey is a 1968 epic science fiction film produced and directed by Stanley Kubrick. The screenplay 
# was written by Kubrick and Arthur C. Clarke, and was inspired by Clarke's short story "The Sentinel" and other 
# short stories by Clarke. A novelisation of the film released after the film's premiere was in part written 
# concurrently with the screenplay. The film, which follows a voyage to Jupiter with the sentient computer HAL after 
# the discovery of a featureless alien monolith affecting human evolution, deals with themes of existentialism, 
# human evolution, technology, artificial intelligence, and the possibility of extraterrestrial life.

```
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

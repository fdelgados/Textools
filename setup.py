from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="texcptulz",
    packages=['txtools',],
    version="0.6.3",
    author="Cisco Delgado",
    author_email="fdelgados@gmail.com",
    description="Tools for cleaning and preprocessing text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fdelgados/Textools.git",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=['nltk', 'scikit-learn', 'spacy', 'spacy-langdetect', 'gensim', 'numpy']
)

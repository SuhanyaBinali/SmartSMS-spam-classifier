##cleanText() function in this file 

import nltk

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords



ps = PorterStemmer()


def clean_text(text):
    text = str(text).lower()

    tokens = nltk.word_tokenize(text)

    filtered_tokens = []

    for token in tokens:
        if token.isalnum():
            filtered_tokens.append(token)

    filtered_tokens_no_stopwords = []

    for token in filtered_tokens:
        if token not in stopwords.words('english'):
            filtered_tokens_no_stopwords.append(token)

    stemmed_tokens = []

    for token in filtered_tokens_no_stopwords:
        stemmed_tokens.append(ps.stem(token))

    return " ".join(stemmed_tokens)
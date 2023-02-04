"""This module holds logic related the preprocessing of the text samples."""

import re
import string
from typing import List

from bulstem.stem import BulStemmer
from lemmagen3 import Lemmatizer
from nltk import ngrams
from stop_words import get_stop_words



def tokenize(raw_text: str, level: str) -> List[str]:
    """Returns numeric representation (i.e. embedding) of a string.

    Args:
        raw_text (str): The text to be embedded.
        level (str): The amount of preprocessing done on the text. Has to be one of `soft`, `medium`, or `hard`.

    Raises:
        NotImplementedError: If the level is not one of `soft`, `medium`, or `hard`.

    Returns:
        List[str]: The embedding of the text.
    """

    STEM_RULES = './stem_rules_context_2_utf8.txt'

    stop_words = get_stop_words('bulgarian')
    lemmatizer = Lemmatizer('bg')
    stemmer = BulStemmer.from_file(STEM_RULES, min_freq=2, left_context=2)

    if level not in {'soft', 'medium', 'hard'}:
        raise NotImplementedError(f'Level {level} is not supported')

    text = raw_text.lower()

    if level in {'hard'}:
        # Remove punctuation.
        text = text.translate(text.maketrans('', '', string.punctuation))
        text = re.sub(r'[a-zA-Z]', "", text)  # Remove non-bulgarian words.

        tokens = text.split()  # Split on whitespace
        tokens = [token for token in tokens if token not in stop_words  # Filter out stopwords
                  and all(c.isalpha() for c in token)]  # and non-word tokens.
    else:
        # Split on punctuation and digits and keep them.
        tokens = re.findall(r"[\w']+|[^\w\s]", text)

    if level in {'medium', 'hard'}:
        # Now: ['песни', 'македония', 'българският', 'бог', ..
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if level in {'hard'}:
        # Now: ['песен', 'македония', 'български', 'бог', ..
        tokens = [stemmer.stem(token) for token in tokens]
        # Now: ['песен', 'македони', 'българск', 'бог', ..

    n_grams = list(ngrams(tokens, 2)) + \
        list(ngrams(tokens, 3)) + list(ngrams(tokens, 4))
    tokens += map(lambda ngrms: ' '.join(ngrms), n_grams)

    return tokens


if __name__ == '__main__':
    print('Hello from preprocessing.py!')

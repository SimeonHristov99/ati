"""Holds the logic for preparing an input string written by the user to a model-ready dataframe."""

import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import tokenize

def load_vectorizer(filename: str) -> TfidfVectorizer:
    """Reads pickled vectorizer into memory.

    Args:
        filename (str): Relative or full path to the pickled vectorizer.
            Must include the extension `.pkl`.

    Returns:
        TfidfVectorizer: An instance of the vectorizer loaded into memory.
    """
    with open(filename, 'rb') as file:
        vectorizer = pickle.load(file)

    return vectorizer


def pipeline_tf_idf(raw_input: str, vectorizer_name: str = '04_vectorizer_hard',
                    tokenization_level: str = 'hard') -> pd.DataFrame:
    """Runs a user-entered string through a pipeline, producing a dataframe ready for modelling.

    Args:
        raw_input (str): The text that is going to be passed through the pipeline.
        vectorizer_name (str, optional): Name of the file from which to load
            the vectorizer. Extension should not be present, but is expected to be `.pkl`.
            Defaults to '04_vectorizer_hard'.
        tokenization_level (str, optional): Level of tokenization. One of `soft`, `medium`, `hard`.
        Defaults to 'hard'.

    Returns:
        pd.DataFrame: A one-dimensional numeric dataframe
            with columns being features from the vectorizer.
    """
    DATA_PATH_PREP = '../DATA/prepared'
    filename = f'{DATA_PATH_PREP}/{vectorizer_name}.pkl'
    vectorizer = load_vectorizer(filename)

    tokens = tokenize(raw_input, tokenization_level)

    text_input_df_vect = pd.DataFrame(
        vectorizer.transform([tokens]).toarray(),
        columns=vectorizer.get_feature_names())

    return text_input_df_vect

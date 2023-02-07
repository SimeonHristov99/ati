"""Holds the logic for preparing an input string written by the user to a model-ready dataframe."""

import json
import numpy as np
import pickle
from collections import Counter
from typing import Tuple

import classla
import pandas as pd
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocessing import tokenize
from readability import (automated_readability_index, coleman_liau_index,
                         flesch_reading_ease, gunning_fog_index, smog)

nlp = classla.Pipeline('bg', processors='tokenize,pos')
model_bg = SentenceTransformer('distiluse-base-multilingual-cased-v2')

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


def pipeline_text_features(raw_input: str, return_tags: bool = False) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    filename = '../DATA/prepared/05_EDA02_text_feats_config.json'
    with open(filename, encoding='UTF-8') as json_file:
        config = json.load(json_file)

    ######################################################
    # Character-based lexical features
    ######################################################
    char_counts = {char: [raw_input.count(char)]
                   for char in config['special_chars']}
    text_features = pd.DataFrame(char_counts)

    ######################################################
    # Sentence- and word-based features
    ######################################################
    tokenized = nlp(raw_input).to_dict()
    parsed_infos = sum([parsed_info for parsed_info, _ in tokenized], [])

    pos_tags = [entry['upos'] for entry in parsed_infos]
    tag_counts = Counter(pos_tags)
    total_tags = len(pos_tags)
    tag_distribution = {tag: count / total_tags
                        for tag, count in tag_counts.items()}

    df_tags = pd.DataFrame(columns=config['POS_Tags'])
    text_features = pd.concat([text_features, df_tags], axis=1)
    for tag, val in tag_distribution.items():
        text_features[tag] = val

    ######################################################
    # Distribution of Token Length
    ######################################################
    token_distribution = Counter(
        [len(entry['text']) for entry in parsed_infos])
    total_tokens = len(token_distribution)
    token_distribution = {length: frequency / total_tokens
                          for length, frequency in token_distribution.items()}

    df_tokens = pd.DataFrame(columns=config['token_lens'])
    text_features = pd.concat([text_features, df_tokens], axis=1)
    for tag, val in token_distribution.items():
        text_features[tag] = val

    ######################################################
    # Distribution of Sentence Length
    ######################################################
    sentence_lengths = [len(sent.split('=')[-1].strip())
                        for _, sent in tokenized]
    length_counts = Counter(sentence_lengths)
    total_sentences = len(sentence_lengths)
    sent_len_distribution = {length: frequency / total_sentences
                             for length, frequency in length_counts.items()
                             if length in config['sent_lens']}

    df_sent_lens = pd.DataFrame(columns=config['sent_lens'])
    text_features = pd.concat([text_features, df_sent_lens], axis=1)
    for tag, val in sent_len_distribution.items():
        text_features[tag] = val

    ######################################################
    # Average Word Length
    ######################################################
    words = raw_input.split()
    total_length = sum(len(word) for word in words)
    text_features['avg_word_len'] = total_length / len(words)

    ######################################################
    # Words in all-caps
    ######################################################
    all_caps_words = [word for word in words if word.isupper()]
    text_features['perc_all_caps'] = len(all_caps_words) / len(words)

    ######################################################
    # Counts of words above and below 2-3 and 6 characters
    ######################################################
    text_features['below2'] = len([word for word in words if len(word) < 2])
    text_features['below3'] = len([word for word in words if len(word) < 3])
    text_features['below6'] = len([word for word in words if len(word) < 6])

    ######################################################
    # Function words
    ######################################################
    df_func_words = pd.DataFrame(columns=config['func_words'])
    function_words = [
        token for token in words if token in config['func_words']]
    fdist = dict(FreqDist(function_words))
    for k, v in fdist.items():
        df_func_words[k] = v
    text_features = pd.concat([text_features, df_func_words], axis=1)

    text_features['fre'] = flesch_reading_ease(raw_input)
    text_features['air'] = automated_readability_index(raw_input)
    text_features['gfi'] = gunning_fog_index(raw_input)
    text_features['cli'] = coleman_liau_index(raw_input)
    text_features['smog'] = smog(raw_input)

    text_features.fillna(0, inplace=True)

    if return_tags:
        df_tags = text_features[config['POS_Tags']].fillna(0)
        df_tags = df_tags.loc[:, (df_tags != 0).any(axis=0)]
        df_tags = df_tags.applymap(lambda val: f'{round(val * 100)}%')
        return (text_features, df_tags)

    return text_features


def pipeline_sbert(raw_input: str):
    NUM_COLS = 13311
    embeddings = model_bg.encode(sent_tokenize(raw_input.strip()))
    embeddings = embeddings.reshape(-1)

    if embeddings.shape[0] > NUM_COLS:
        embeddings = embeddings[:NUM_COLS+1]
    else:
        embeddings = np.append(embeddings, np.zeros(NUM_COLS - len(embeddings) + 1))

    return pd.DataFrame(embeddings).T


if __name__ == '__main__':
    raw_input = 'Алеко Константинов е роден в Свищов. Кой е Иван Вазов?'
    sbert_embeds = pipeline_sbert(raw_input)
    print(sbert_embeds)
    print(sbert_embeds.shape)

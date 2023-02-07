"""Holds formulas for calculating various readability indices."""

import numpy as np


def count_syllables(word):
    vowels = 'аъоуеиАЪОУЕИ'
    syllables = 0
    for i in range(len(word)):
        if word[i] in vowels and (i == 0 or word[i-1] not in vowels):
            syllables += 1
    return syllables


def flesch_reading_ease(text):
    words = text.split()
    num_words = len(words)
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    num_syllables = sum([count_syllables(word) for word in words])

    if num_sentences == 0 or num_words == 0:
        return 0

    fre = 206.835 - (1.015 * (num_words / num_sentences)) - \
        (84.6 * (num_syllables / num_words))
    fre = np.clip(fre, 0, 100)
    return fre


def automated_readability_index(text):
    words = text.split()
    num_characters = sum(len(word) for word in words)
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    num_words = len(words)

    if num_words == 0 or num_sentences == 0:
        return 15

    ari = 4.71 * (num_characters / num_words) + 0.5 * \
        (num_words / num_sentences) - 21.43
    ari = np.clip(ari, 1, 14)
    return ari


def gunning_fog_index(text):
    words = text.split()
    complex_words = sum(1 for word in words if len(word.split('-')) > 1)
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    word_count = len(words)

    if sentence_count == 0 or word_count == 0:
        return 18

    gfi = 0.4 * ((word_count / sentence_count) +
                 100 * (complex_words / word_count))
    gfi = np.clip(gfi, 6, 17)
    return gfi


def coleman_liau_index(text):
    num_characters = sum(len(word) for word in text.split())
    num_words = len(text.split())
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    L = num_characters / num_words * 100
    S = num_sentences / num_words * 100
    cli = 0.0588 * L - 0.296 * S - 15.8
    return cli


def smog(text):
    sentences = text.split('.')
    polysyllables = 0
    for sentence in sentences:
        words = sentence.split()
        for word in words:
            syllables = len([char for char in word if char in 'аъоуеиАЪОУЕИ'])
            if syllables >= 3:
                polysyllables += 1
    smog = 1.043 * np.sqrt(polysyllables * (30 / len(sentences))) + 3.1291
    return round(smog, 1)

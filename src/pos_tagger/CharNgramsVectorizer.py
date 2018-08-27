from collections import Counter
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
import numpy as np

from pos_tagger.utils import flatten, get_ngrams


class CharNgramsVectorizer:
    """
    Character level ngrams vectorizer
    """

    PAD = '<p>'
    UNK = '<u>'

    def __init__(self):
        """

        """
        self.tag2id = None
        self.id2tag = None
        self.token2id = None
        self.id2token = None

    def fit_transform(self, words_corpus, labels_corpus, ngrams, min_tokens, return_data, seq_maxlen):
        """
        :param words_corpus: corpus with words (corpus is split on sentences):
            [[word_1_1, word_1_2, ... ], [word_2_1, word_2_2, ...], ...]
        :param labels_corpus: corpus with labels (the same size as the words_corpus), but now we have not single words,
            but list with labels for each word
            [[[labs for word_1_1], [labs for word_1_2], ... ], [[labs for word_2_1], [labs for word_2_2], ...], ...]
        :param ngrams: character ngrams len (2 is okay)
        :param min_tokens: trim your labels classes dict (and also the output dim) by removing rare classes which have
            less than min_tokens number
        :param return_data: if False, model will just fit tag2id and token2id dicts and will not return training data
        :param seq_maxlen: delete samples which exceed maximum length (which is measured in number of char ngrams) from
            the training data.
        :return: data which is fully prepared for model training: X_left, X_right, X_word, Y
        """
        features_counter = Counter(flatten(flatten(labels_corpus)))
        unique_features = sorted(list(features_counter.keys()))
        self.tag2id = {unique_features[i]: i for i in range(len(unique_features))}
        self.id2tag = {i: unique_features[i] for i in range(len(unique_features))}

        X_left, X_right, X_word, Y = self.vectorize(words_corpus=words_corpus, ngrams=ngrams,
                                                    labels_corpus=labels_corpus, to_id=False)

        add_chars = [self.PAD, self.UNK]
        chars_counter = Counter(np.hstack(X_left))
        unique_chars = sorted([x[0] for x in chars_counter.items() if x[1] >= min_tokens] + add_chars)
        self.token2id = {unique_chars[i]: i + len(add_chars) for i in range(len(unique_chars))}

        for i in range(len(add_chars)):
            self.token2id[add_chars[i]] = i

        self.id2token = {self.token2id[x]: x for x in self.token2id}

        if seq_maxlen:
            mask = np.array([len(x) <= seq_maxlen for x in X_left])
            mask *= np.array([len(x) <= seq_maxlen for x in X_right])
            mask *= np.array([len(x) <= seq_maxlen for x in X_word])
        else:
            mask = np.ones(len(X_left))

        if return_data:
            X_left = [[self.token2id.get(x, self.token2id[self.UNK]) for x in y] for y in np.array(X_left)[mask]]
            X_right = [[self.token2id.get(x, self.token2id[self.UNK]) for x in y] for y in np.array(X_right)[mask]]
            X_word = [[self.token2id.get(x, self.token2id[self.UNK]) for x in y] for y in np.array(X_word)[mask]]

            X_left = pad_sequences(X_left, value=self.token2id[self.PAD], padding='pre')
            X_right = pad_sequences(X_right, value=self.token2id[self.PAD], padding='post')
            X_word = pad_sequences(X_word, value=self.token2id[self.PAD], padding='post')
            Y = np.vstack([to_categorical(x, len(self.tag2id)).any(axis=0) for x in np.array(Y)[mask]]).astype(np.int32)

            return X_left, X_right, X_word, Y

    def vectorize(self, words_corpus, ngrams, to_id=False, labels_corpus=None):
        """
        Vectorize your corpus. Padding will not be applied, because one may want to predict the sequences as is,
            with variable length
        :param words_corpus: corpus with words (corpus is split on sentences):
            [[word_1_1, word_1_2, ... ], [word_2_1, word_2_2, ...], ...]
        :param ngrams: see fit_transform method
        :param to_id: if True, transform tokens to their indexes (vectorizer must be fitted)
        :param labels_corpus: corpus with labels (the same size as the words_corpus), but now we have not single words,
            but list with labels for each word
            [[[labs for word_1_1], [labs for word_1_2], ... ], [[labs for word_2_1], [labs for word_2_2], ...], ...]
            (vectorizer must be fitted)
        :return: X_left, X_right, X_word, Y
        """
        if to_id:
            assert self.token2id is not None
        if labels_corpus:
            assert self.tag2id is not None

        X_left = []
        X_right = []
        X_word = []
        Y = []

        for i in range(len(words_corpus)):
            words = words_corpus[i]

            for j in range(len(words)):
                left_words = ' '.join([x for x in words[:j]])
                right_words = ' '.join([x for x in words[j + 1:]])
                word = words[j]

                x_left = get_ngrams(' ' + left_words + ' ' + word + ' ', n=ngrams)
                x_right = get_ngrams(' ' + word + ' ' + right_words + ' ', n=ngrams)
                x_word = get_ngrams(' ' + word + ' ', n=ngrams)

                if to_id:
                    x_left = [self.token2id.get(x, self.token2id[self.UNK]) for x in x_left]
                    x_right = [self.token2id.get(x, self.token2id[self.UNK]) for x in x_right]
                    x_word = [self.token2id.get(x, self.token2id[self.UNK]) for x in x_word]

                X_left.append(np.array(x_left))
                X_right.append(np.array(x_right))
                X_word.append(np.array(x_word))

                if labels_corpus:
                    word_tags = labels_corpus[i][j]
                    y_inds = [self.tag2id[x] for x in word_tags]
                    Y.append(y_inds)

        return X_left, X_right, X_word, Y

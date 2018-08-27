from keras.layers import *
from keras import Model
from keras.optimizers import *
from keras.models import load_model


class POSTaggerModel:
    """
    This is the multi-class/multi-label mode which is oriented on part of speech tagging task
    """

    def __init__(self, model_path=None, emb_dim=None, hidden_dim=None, input_dim=None, out_dim=None, lr=None):
        """
        :param model_path: str, path to the pre-trained mode. If set, all other parameters will be ignored
        :param emb_dim: input embedding size
        :param hidden_dim: hidden lstm size
        :param input_dim: input size (size of your vocabulary, where 0 index is reserved for padding)
        :param out_dim: output size (number of classes)
        :param lr: learning rate for Adagrad optimizer
        """

        if model_path is not None:
            self.__model = load_model(model_path)
        else:
            embedding_layer = Embedding(input_dim=input_dim, output_dim=emb_dim, mask_zero=True)
            rnn_left_layer = Bidirectional(LSTM(units=hidden_dim, return_sequences=False))
            rnn_right_layer = Bidirectional(LSTM(units=hidden_dim, return_sequences=False))
            rnn_word_layer = Bidirectional(LSTM(units=hidden_dim, return_sequences=False))
            output_layer = Dense(units=out_dim, activation='sigmoid')

            input_left = Input(shape=(None,))
            input_right = Input(shape=(None,))
            input_word = Input(shape=(None,))

            embedding_left = embedding_layer(input_left)
            embedding_right = embedding_layer(input_right)
            embedding_word = embedding_layer(input_word)

            rnn_left = rnn_left_layer(embedding_left)
            rnn_right = rnn_right_layer(embedding_right)
            rnn_word = rnn_word_layer(embedding_word)
            rnn_concat = concatenate([rnn_left, rnn_right, rnn_word])

            output = output_layer(rnn_concat)

            optimizer = Adagrad(lr)
            model = Model([input_left, input_right, input_word], [output])
            model.compile(optimizer=optimizer, loss='binary_crossentropy')

            self.__model = model

    def fit(self, X, Y, epochs, batch_size, validation_data, callbacks):
        """
        :param X: list with 3 X-arrays
        :param Y: Y array
        :param epochs: number of epochs to train
        :param batch_size: batch size
        :param validation_data: tuple with: [list of 3 X-arrays, Y]
        :param callbacks: Keras callbacks to the model
        """

        self.__model.fit(x=X, y=Y, batch_size=batch_size, epochs=epochs, validation_data=validation_data,
                         callbacks=callbacks)

    def predict(self, X):
        return self.__model.predict(x=X)

    def save_model(self, path):
        self.__model.save(path)

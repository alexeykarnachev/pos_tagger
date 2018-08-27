from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from pos_tagger.CharNgramsVectorizer import CharNgramsVectorizer
from pos_tagger.POSTaggerModel import POSTaggerModel
from pos_tagger.utils import parse_opencorpora

OC_XML = '../data/annot.opcorpora.no_ambig.xml'
MODEL_PATH = "../model/pos_model-{epoch:02d}-{val_loss:.5f}.hdf5"

if __name__ == '__main__':
    words_corpus, labels_corpus = parse_opencorpora(OC_XML)
    vectorizer = CharNgramsVectorizer()
    X_left, X_right, X_word, Y = vectorizer.fit_transform(words_corpus=words_corpus, labels_corpus=labels_corpus,
                                                          ngrams=2, min_tokens=5, return_data=True, seq_maxlen=400)

    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)

    X_left_train, X_left_test, X_right_train, X_right_test, X_word_train, X_word_test, Y_train, Y_test = \
        train_test_split(X_left, X_right, X_word, Y, test_size=0.03)

    model = POSTaggerModel(model_path=None, emb_dim=100, hidden_dim=100, input_dim=len(vectorizer.id2token),
                           out_dim=len(vectorizer.id2tag), lr=0.05)

    model.fit(X=[X_left_train, X_right_train, X_word_train], Y=Y_train, epochs=7, batch_size=128,
              validation_data=([X_left_test, X_right_test, X_word_test], Y_test), callbacks=[checkpoint])
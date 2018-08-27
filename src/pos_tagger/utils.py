from xml.dom import minidom


def flatten(list_):
    return [item for sublist in list_ for item in sublist]


def get_ngrams(str_, n):
    return [str_[i:i + n] for i in range(len(str_) - n + 1)]


def parse_opencorpora(xml_path):
    mydoc = minidom.parse(xml_path)
    sents = mydoc.getElementsByTagName('sentence')

    WORDS = []
    FEATURES = []

    for sent in sents:

        tokens = sent.getElementsByTagName('token')

        words = []
        features = []

        for token in tokens:
            token_text = token.getAttribute('text').lower()
            g_ = token.getElementsByTagName('g')
            token_features = flatten([[t[1] for t in x.attributes.items()] for x in g_])
            words.append(token_text)
            features.append(token_features)

        assert len(words) == len(features)

        WORDS.append(words)
        FEATURES.append(features)

    return WORDS, FEATURES
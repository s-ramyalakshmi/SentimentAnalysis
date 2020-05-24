import tensorflow
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
import numpy as np

lemmatizer = WordNetLemmatizer()
loxi = None

def create_lexicon(pos, neg):
    lexicon = []
    for sample in [pos, neg]:
        for text in sample:
            all_words = word_tokenize(text.lower())
            lexicon.extend(all_words)

    lex = []
    for word in lexicon:
        lex.append(lemmatizer.lemmatize(word))

    w_count = Counter(lex)
    words = []

    for w in w_count:
        if 7000 > w_count[w] > 10:
            words.append(w)

    print(len(words), '\n')
    return words

def create_sample(sample, lexicon, classificatin):
    featureset = []
    words = []
    for text in sample:
        curr_words = word_tokenize(text.lower())
        words = [lemmatizer.lemmatize(i) for i in curr_words]
        features = np.zeros(len(lexicon))
        for word in words:
            if word.lower() in lexicon:
                index = lexicon.index(word.lower())
                features[index] += 1
        features = list(features)
        featureset.append([features, classificatin])
        break
    return featureset


def create_featureset_and_labels(train_pos, train_neg, test_pos, test_neg):
    global loxi

    pos = []
    pos.extend(test_pos)
    pos.extend(train_pos)

    neg = []
#    neg.extend(train_neg)
#    neg.extend(test_neg)

    print('loxi call')
    loxi = create_lexicon(pos, neg)
    print('loxi end')
#    features = []

    print('sample call')
    train_x = create_sample(train_pos, loxi, [1, 0])
    print('sample end')
#    train_y = create_sample(train_neg, loxi, [0, 1])
#    test_x = create_sample(test_pos, loxi, [1, 0])
#    test_y = create_sample(test_neg, loxi, [0, 1])

#    print(train_x[0:10])

    






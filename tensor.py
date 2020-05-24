import tensorflow as tf
import os
from process import create_featureset_and_labels

#train data

train_pos = []
path = '/Users/ramyalakshmi.s/Documents/Project/ML/Sentiment-Analysis/aclImdb/train/pos/'
for fileName in os.listdir(path):
    if(fileName.endswith('.txt')):
        with open (path+fileName) as myfile:
            data = myfile.readlines()
            train_pos.extend(data)

train_neg = []
path = '/Users/ramyalakshmi.s/Documents/Project/ML/Sentiment-Analysis/aclImdb/train/neg/'
for fileName in os.listdir(path):
    if(fileName.endswith('.txt')):
        with open (path+fileName) as myfile:
            data = myfile.readlines()
            train_neg.extend(data)

test_pos = []
path = '/Users/ramyalakshmi.s/Documents/Project/ML/Sentiment-Analysis/aclImdb/test/pos/'
for fileName in os.listdir(path):
    if(fileName.endswith('.txt')):
        with open (path+fileName) as myfile:
            data = myfile.readlines()
            test_pos.extend(data)

test_neg = []
path = '/Users/ramyalakshmi.s/Documents/Project/ML/Sentiment-Analysis/aclImdb/test/neg/'
for fileName in os.listdir(path):
    if(fileName.endswith('.txt')):
        with open (path+fileName) as myfile:
            data = myfile.readlines()
            test_neg.extend(data)


print("call")
create_featureset_and_labels(train_pos, train_neg, test_pos, test_neg)   
print("off")      
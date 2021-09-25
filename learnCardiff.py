import time
import json
import numpy as np
import pandas as pd
import spacy

from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

nlp = spacy.load("en_core_web_lg")

dataframe = pd.read_json("data.json")

# Code has been partially adapted from the following source example (Accessed 10/03/2021): https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html
data_traintune, data_test = train_test_split(dataframe, test_size=0.2 ,random_state=42)
data_train, data_tune = train_test_split(data_traintune, test_size=0.1 ,random_state=42)

# Word vector code adapted from the following source (Accessed 11/03/2021): https://lvngd.com/blog/spacy-word-vectors-as-features-in-scikit-learn/
# Converting data into word vectors
data_list = [nlp(doc).vector.reshape(1,-1) for doc in data_tune.post]
data_tune_x_wvec = np.concatenate(data_list)
data_tune_y = data_tune.label

data_list = [nlp(doc).vector.reshape(1,-1) for doc in data_train.post]
data_train_x_wvec = np.concatenate(data_list)
data_train_y = data_train.label

data_list = [nlp(doc).vector.reshape(1,-1) for doc in data_test.post]
data_test_x_wvec = np.concatenate(data_list)
data_test_y = data_test.label

# Tuning and training the LinearSVC classifier
print("LinearSVC")
Cs = [.01, .1, 1, 10, 100]
bestScore = 0
bestParam = ''
for singleC in Cs:
    tuner = LinearSVC(max_iter=100000, C=singleC)
    tuner.fit(data_train_x_wvec, data_train_y)
    predicted = tuner.predict(data_tune_x_wvec)
    f1Score = metrics.f1_score(data_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = singleC

clf = LinearSVC(max_iter=100000, C=bestParam)
clf.fit(data_train_x_wvec, data_train_y)

predicted = clf.predict(data_test_x_wvec)
print(metrics.classification_report(data_test_y, predicted))
print("")

# Tuning and training the Kneighbors classifier
print("KNeighbors")
n_neighbors = [3, 5, 10, 15, 20, 30, 44, 70]
bestScore = 0
bestParam = ''
for n in n_neighbors:
    tuner = KNeighborsClassifier(n_neighbors=n)
    tuner.fit(data_train_x_wvec, data_train_y)
    predicted = tuner.predict(data_tune_x_wvec)
    f1Score = metrics.f1_score(data_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = n

clf = KNeighborsClassifier(n_neighbors=bestParam)
clf.fit(data_train_x_wvec, data_train_y)

predicted = clf.predict(data_test_x_wvec)
print(metrics.classification_report(data_test_y, predicted))
print("")

# Tuning and training the Naive Bayes classifiers
print("BernoulliNB")
bestScore = 0
bestParam = ''
alphas = [.000001, .00001, .0001, .001, .01, .1, 1, 10, 100]
for alpha in alphas:
    tuner = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', BernoulliNB(alpha=alpha))])
    tuner.fit(data_train.post, data_train.label)
    predicted = tuner.predict(data_tune.post)
    f1Score = metrics.f1_score(data_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = alpha

pipeline= Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', BernoulliNB(alpha=bestParam))])
pipeline.fit(data_train.post, data_train.label)

predicted = pipeline.predict(data_test.post)
print(metrics.classification_report(data_test.label, predicted))
print("")

print("MultinomialNB")
bestScore = 0
bestParam = ''
alphas = [.000001, .00001, .0001, .001, .01, .1, 1, 10, 100]
for alpha in alphas:
    tuner = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=alpha))])
    tuner.fit(data_train.post, data_train.label)
    predicted = tuner.predict(data_tune.post)
    f1Score = metrics.f1_score(data_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = alpha

pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=bestParam))])
pipeline.fit(data_train.post, data_train.label)

predicted = pipeline.predict(data_test.post)
print(metrics.classification_report(data_test.label, predicted))
print("")
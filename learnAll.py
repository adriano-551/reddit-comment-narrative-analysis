import json
import numpy as np
import pandas as pd
import spacy

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

nlp = spacy.load("en_core_web_lg")

cardiffData = pd.read_json("cardiffData.json")

# Code has been partially adapted from the following source example (Accessed 10/03/2021): https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html
cardiffData_traintune, cardiffData_test = train_test_split(cardiffData, test_size=0.2 ,random_state=42)
cardiffData_train, cardiffData_tune = train_test_split(cardiffData_traintune, test_size=0.1 ,random_state=42)

# Word vector code adapted from the following source (Accessed 11/03/2021): https://lvngd.com/blog/spacy-word-vectors-as-features-in-scikit-learn/
# Converting data into word vectors
cardiffData_list = [nlp(doc).vector.reshape(1,-1) for doc in cardiffData_tune.post]
cardiffData_tune_x_wvec = np.concatenate(cardiffData_list)
cardiffData_tune_y = cardiffData_tune.label

cardiffData_list = [nlp(doc).vector.reshape(1,-1) for doc in cardiffData_train.post]
cardiffData_train_x_wvec = np.concatenate(cardiffData_list)
cardiffData_train_y = cardiffData_train.label

cardiffData_list = [nlp(doc).vector.reshape(1,-1) for doc in cardiffData_test.post]
cardiffData_test_x_wvec = np.concatenate(cardiffData_list)
cardiffData_test_y = cardiffData_test.label

# Section 1, Testing how accurately a classifier trained using 1000 posts from /r/Cardiff can determine narrative posts
# Tuning and training the LinearSVC classifier
print("Section 1a: Cardiff trained classifier, Cardiff data testing")
print("LinearSVC")
Cs = [.01, .1, 1, 10, 100]
bestScore = 0
bestParam = ''
for singleC in Cs:
    tuner = LinearSVC(max_iter=100000, C=singleC)
    tuner.fit(cardiffData_train_x_wvec, cardiffData_train_y)
    predicted = tuner.predict(cardiffData_tune_x_wvec)
    f1Score = metrics.f1_score(cardiffData_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = singleC

cardiffSVC = LinearSVC(max_iter=100000, C=bestParam)
cardiffSVC.fit(cardiffData_train_x_wvec, cardiffData_train_y)

predicted = cardiffSVC.predict(cardiffData_test_x_wvec)
print(metrics.f1_score(cardiffData_test_y, predicted, pos_label='narrative'))
print("")

# Tuning and training the Kneighbors classifier
print("KNeighbors")
n_neighbors = [3, 5, 10, 15, 20, 30, 44, 70]
bestScore = 0
bestParam = ''
for n in n_neighbors:
    tuner = KNeighborsClassifier(n_neighbors=n)
    tuner.fit(cardiffData_train_x_wvec, cardiffData_train_y)
    predicted = tuner.predict(cardiffData_tune_x_wvec)
    f1Score = metrics.f1_score(cardiffData_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = n

cardiffKNeighbors = KNeighborsClassifier(n_neighbors=bestParam)
cardiffKNeighbors.fit(cardiffData_train_x_wvec, cardiffData_train_y)

predicted = cardiffKNeighbors.predict(cardiffData_test_x_wvec)
print(metrics.f1_score(cardiffData_test_y, predicted, pos_label='narrative'))
print("")

# Tuning and training the Naive Bayes classifiers
print("BernoulliNB")
bestScore = 0
bestParam = ''
alphas = [.000001, .00001, .0001, .001, .01, .1, 1, 10, 100]
for alpha in alphas:
    tuner = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', BernoulliNB(alpha=alpha))])
    tuner.fit(cardiffData_train.post, cardiffData_train.label)
    predicted = tuner.predict(cardiffData_tune.post)
    f1Score = metrics.f1_score(cardiffData_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = alpha

cardiffBernoulli = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', BernoulliNB(alpha=bestParam))])
cardiffBernoulli.fit(cardiffData_train.post, cardiffData_train.label)

predicted = cardiffBernoulli.predict(cardiffData_test.post)
print(metrics.f1_score(cardiffData_test_y, predicted, pos_label='narrative'))
print("")

print("MultinomialNB")
bestScore = 0
bestParam = ''
alphas = [.000001, .00001, .0001, .001, .01, .1, 1, 10, 100]
for alpha in alphas:
    tuner = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=alpha))])
    tuner.fit(cardiffData_train.post, cardiffData_train.label)
    predicted = tuner.predict(cardiffData_tune.post)
    f1Score = metrics.f1_score(cardiffData_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = alpha

cardiffMultinomial = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=bestParam))])
cardiffMultinomial.fit(cardiffData_train.post, cardiffData_train.label)

predicted = cardiffMultinomial.predict(cardiffData_test.post)
print(metrics.f1_score(cardiffData_test_y, predicted, pos_label='narrative'))
print("")

# Section 2 using 1000 posts from other subreddits
# Importing other subreddits data
allSubData = pd.read_json("allSubData.json")

allSubData_traintune, allSubData_test = train_test_split(allSubData, test_size=0.2 ,random_state=42)
allSubData_train, allSubData_tune = train_test_split(allSubData_traintune, test_size=0.1 ,random_state=42)

# Converting data into word vectors
allSubData_list = [nlp(doc).vector.reshape(1,-1) for doc in allSubData_tune.post]
allSubData_tune_x_wvec = np.concatenate(allSubData_list)
allSubData_tune_y = allSubData_tune.label

allSubData_list = [nlp(doc).vector.reshape(1,-1) for doc in allSubData_train.post]
allSubData_train_x_wvec = np.concatenate(allSubData_list)
allSubData_train_y = allSubData_train.label

allSubData_list = [nlp(doc).vector.reshape(1,-1) for doc in allSubData_test.post]
allSubData_test_x_wvec = np.concatenate(allSubData_list)
allSubData_test_y = allSubData_test.label

# Section 1b, testing how well the cardiff classifier can classify reddit posts from other subreddits
print("Section 1b: Cardiff trained classifier, all subreddit testing")
print("LinearSVC")
predicted = cardiffSVC.predict(allSubData_test_x_wvec)
print(metrics.f1_score(allSubData_test_y, predicted, pos_label='narrative'))
print("")

print("KNeighbors")
predicted = cardiffKNeighbors.predict(allSubData_test_x_wvec)
print(metrics.f1_score(allSubData_test_y, predicted, pos_label='narrative'))
print("")

print("BernoulliNB")
predicted = cardiffBernoulli.predict(allSubData_test.post)
print(metrics.f1_score(allSubData_test_y, predicted, pos_label='narrative'))
print("")

print("MultinomialNB")
predicted = cardiffMultinomial.predict(allSubData_test.post)
print(metrics.f1_score(allSubData_test_y, predicted, pos_label='narrative'))
print("")

# Section 2a, Testing how well a classifier trained on 1000 random reddit posts can classify narrative posts
print("Section 2a: All subreddit trained classifier, All Subreddit testing")
# Tuning and training the LinearSVC classifier
print("LinearSVC")
Cs = [.01, .1, 1, 10, 100]
bestScore = 0
bestParam = ''
for singleC in Cs:
    tuner = LinearSVC(max_iter=100000, C=singleC)
    tuner.fit(allSubData_train_x_wvec, allSubData_train_y)
    predicted = tuner.predict(allSubData_tune_x_wvec)
    f1Score = metrics.f1_score(allSubData_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = singleC

allSubSVC = LinearSVC(max_iter=100000, C=bestParam)
allSubSVC.fit(allSubData_train_x_wvec, allSubData_train_y)

predicted = allSubSVC.predict(allSubData_test_x_wvec)
print(metrics.f1_score(allSubData_test_y, predicted, pos_label='narrative'))
print("")

# Tuning and training the Kneighbors classifier
print("KNeighbors")
n_neighbors = [3, 5, 10, 15, 20, 30, 44, 70]
bestScore = 0
bestParam = ''
for n in n_neighbors:
    tuner = KNeighborsClassifier(n_neighbors=n)
    tuner.fit(allSubData_train_x_wvec, allSubData_train_y)
    predicted = tuner.predict(allSubData_tune_x_wvec)
    f1Score = metrics.f1_score(allSubData_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = n

allSubKNeighbors = KNeighborsClassifier(n_neighbors=bestParam)
allSubKNeighbors.fit(allSubData_train_x_wvec, allSubData_train_y)

predicted = allSubKNeighbors.predict(allSubData_test_x_wvec)
print(metrics.f1_score(allSubData_test_y, predicted, pos_label='narrative'))
print("")

# Tuning and training the Naive Bayes classifiers
print("BernoulliNB")
bestScore = 0
bestParam = ''
alphas = [.000001, .00001, .0001, .001, .01, .1, 1, 10, 100]
for alpha in alphas:
    tuner = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', BernoulliNB(alpha=alpha))])
    tuner.fit(allSubData_train.post, allSubData_train.label)
    predicted = tuner.predict(allSubData_tune.post)
    f1Score = metrics.f1_score(allSubData_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = alpha

allSubBernoulli = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', BernoulliNB(alpha=bestParam))])
allSubBernoulli.fit(allSubData_train.post, allSubData_train.label)

predicted = allSubBernoulli.predict(allSubData_test.post)
print(metrics.f1_score(allSubData_test_y, predicted, pos_label='narrative'))
print("")

print("MultinomialNB")
bestScore = 0
bestParam = ''
alphas = [.000001, .00001, .0001, .001, .01, .1, 1, 10, 100]
for alpha in alphas:
    tuner = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=alpha))])
    tuner.fit(allSubData_train.post, allSubData_train.label)
    predicted = tuner.predict(allSubData_tune.post)
    f1Score = metrics.f1_score(allSubData_tune_y, predicted, pos_label='narrative')

    if f1Score > bestScore:
        bestScore = f1Score
        bestParam = alpha

allSubMultinomial = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=bestParam))])
allSubMultinomial.fit(allSubData_train.post, allSubData_train.label)

predicted = allSubMultinomial.predict(allSubData_test.post)
print(metrics.f1_score(allSubData_test_y, predicted, pos_label='narrative'))
print("")

# Section 2b, testing how well a classifier trained on 1000 random reddit posts can classify cardiff reddit posts
print("Section 2b: All subreddit trained classifier, Cardiff data testing")
print("LinearSVC")
predicted = allSubSVC.predict(cardiffData_test_x_wvec)
print(metrics.f1_score(cardiffData_test_y, predicted, pos_label='narrative'))
print("")

print("KNeighbors")
predicted = allSubKNeighbors.predict(cardiffData_test_x_wvec)
print(metrics.f1_score(cardiffData_test_y, predicted, pos_label='narrative'))
print("")

print("BernoulliNB")
predicted = allSubBernoulli.predict(cardiffData_test.post)
print(metrics.f1_score(cardiffData_test_y, predicted, pos_label='narrative'))
print("")

print("MultinomialNB")
predicted = allSubMultinomial.predict(cardiffData_test.post)
print(metrics.f1_score(cardiffData_test_y, predicted, pos_label='narrative'))
print("")

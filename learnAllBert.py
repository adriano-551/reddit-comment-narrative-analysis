import json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics

from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging

# Code is placed into a run() method in order to prevent a possible bug - sourced from github: https://github.com/ThilinaRajapakse/simpletransformers/issues/225
def run():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # Grabbing BERT data seperately as the format for the BERT classifier is different
    cardiffDataBert = pd.read_json("cardiffDataBert.json")
    cardiffDataBert_traintune, cardiffDataBert_test = train_test_split(cardiffDataBert, test_size=0.2 ,random_state=42)
    cardiffDataBert_train, cardiffDataBert_tune = train_test_split(cardiffDataBert_traintune, test_size=0.1 ,random_state=42)

    cardiffDataBert_test.columns = ["text", "labels"]
    cardiffDataBert_tune.columns = ["text", "labels"]
    cardiffDataBert_train.columns = ["text", "labels"]

    cardiffDataBert_tune_text = cardiffDataBert_tune.text.values.tolist()
    cardiffDataBert_test_text = cardiffDataBert_test.text.values.tolist()
    
    # Section 1, Testing how accurately a classifier trained using 1000 posts from /r/Cardiff can determine narrative posts
    print("Section 1a: Cardiff trained classifier")
    print("BERT")
    bestScore = 0
    bestParams = ""
    learningRates = [2e-5, 3e-5, 4e-5, 5e-5]
    for learningRate in learningRates:
        model_args = ClassificationArgs()
        model_args.learning_rate = learningRate
        model_args.num_train_epochs = 4
        model_args.use_multiprocessing = False
        model_args.overwrite_output_dir = True
        tuner = ClassificationModel(model_type="roberta", model_name="roberta-base", args=model_args, use_cuda=False)
        tuner.train_model(cardiffDataBert_train)
        predicted, raw = tuner.predict(cardiffDataBert_tune_text)
        
        f1Score = metrics.f1_score(cardiffDataBert_tune.labels, predicted)

        if f1Score > bestScore:
            bestScore = f1Score
            bestParams = learningRate

    model_args = ClassificationArgs()
    model_args.learning_rate = bestParams
    model_args.num_train_epochs = 4
    model_args.use_multiprocessing = False
    model_args.overwrite_output_dir = True
    cardiffBert = ClassificationModel(model_type="roberta", model_name="roberta-base", args=model_args, use_cuda=False)
    cardiffBert.train_model(cardiffDataBert_train)

    predicted, raw = cardiffBert.predict(cardiffDataBert_test_text)
    print(metrics.f1_score(cardiffDataBert_test.labels, predicted))
    print("")
    
    # Section 2 using 1000 posts from other subreddits
    #Gathering data for use with BERT classifier
    allSubDataBert = pd.read_json("allSubDataBert.json")
    allSubDataBert_traintune, allSubDataBert_test = train_test_split(allSubDataBert, test_size=0.2 ,random_state=42)
    allSubDataBert_train, allSubDataBert_tune = train_test_split(allSubDataBert_traintune, test_size=0.1 ,random_state=42)

    allSubDataBert_test.columns = ["text", "labels"]
    allSubDataBert_tune.columns = ["text", "labels"]
    allSubDataBert_train.columns = ["text", "labels"]

    allSubDataBert_tune_text = allSubDataBert_tune.text.values.tolist()
    allSubDataBert_test_text = allSubDataBert_test.text.values.tolist()
    
    # Section 2a, testing how well the cardiff classifier can classify reddit posts from other subreddits
    print("Section 1b: Cardiff trained classifier, all subreddit testing")
    print("BERT")
    predicted, raw = cardiffBert.predict(allSubDataBert_test_text)
    print(metrics.f1_score(allSubDataBert_test.labels, predicted))
    print("")

    # Section 2b, Testing how well a classifier trained on 1000 random reddit posts can classify narrative posts
    print("Section 2a: All subreddit trained classifier, All Subreddit testing")
    print("BERT")
    bestScore = 0
    bestParams = ""
    learningRates = [2e-5, 3e-5, 4e-5, 5e-5]
    for learningRate in learningRates:
        model_args = ClassificationArgs()
        model_args.learning_rate = learningRate
        model_args.num_train_epochs = 4
        model_args.use_multiprocessing = False
        model_args.overwrite_output_dir = True
        tuner = ClassificationModel(model_type="roberta", model_name="roberta-base", args=model_args, use_cuda=False)
        tuner.train_model(allSubDataBert_train)
        predicted, raw = tuner.predict(allSubDataBert_tune_text)
        
        f1Score = metrics.f1_score(allSubDataBert_tune.labels, predicted)

        if f1Score > bestScore:
            bestScore = f1Score
            bestParams = learningRate

    model_args = ClassificationArgs()
    model_args.learning_rate = bestParams
    model_args.num_train_epochs = 4
    model_args.use_multiprocessing = False
    model_args.overwrite_output_dir = True
    allSubBert = ClassificationModel(model_type="roberta", model_name="roberta-base", args=model_args, use_cuda=False)
    allSubBert.train_model(allSubDataBert_train)

    predicted, raw = allSubBert.predict(allSubDataBert_test_text)
    print(metrics.f1_score(allSubDataBert_test.labels, predicted))
    print("")

    # Section 2c, testing how well a classifier trained on 1000 random reddit posts can classify cardiff reddit posts
    print("Section 2b: All subreddit trained classifier, Cardiff data testing")
    print("BERT")
    predicted, raw = allSubBert.predict(cardiffDataBert_test_text)
    print(metrics.f1_score(cardiffDataBert_test.labels, predicted))
    print("")

if __name__ == '__main__':
    run()
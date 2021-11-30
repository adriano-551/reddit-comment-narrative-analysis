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
    cardiff_data_bert = pd.read_json("cardiff_data_bert.json")
    cardiff_data_bert_traintune, cardiff_data_bert_test = train_test_split(cardiff_data_bert, test_size=0.2 ,random_state=42)
    cardiff_data_bert_train, cardiff_data_bert_tune = train_test_split(cardiff_data_bert_traintune, test_size=0.1 ,random_state=42)

    cardiff_data_bert_test.columns = ["text", "labels"]
    cardiff_data_bert_tune.columns = ["text", "labels"]
    cardiff_data_bert_train.columns = ["text", "labels"]

    cardiff_data_bert_tune_text = cardiff_data_bert_tune.text.values.tolist()
    cardiff_data_bert_test_text = cardiff_data_bert_test.text.values.tolist()
    
    # Section 1, Testing how accurately a classifier trained using 1000 posts from /r/Cardiff can determine narrative posts
    print("Section 1a: Cardiff trained classifier")
    print("BERT")
    best_score = 0
    best_params = ""
    learning_rates = [2e-5, 3e-5, 4e-5, 5e-5]
    for learning_rate in learning_rates:
        model_args = ClassificationArgs()
        model_args.learning_rate = learning_rate
        model_args.num_train_epochs = 4
        model_args.use_multiprocessing = False
        model_args.overwrite_output_dir = True
        tuner = ClassificationModel(model_type="roberta", model_name="roberta-base", args=model_args, use_cuda=False)
        tuner.train_model(cardiff_data_bert_train)
        predicted, raw = tuner.predict(cardiff_data_bert_tune_text)
        
        f1_score = metrics.f1_score(cardiff_data_bert_tune.labels, predicted)

        if f1_score > best_score:
            best_score = f1_score
            best_params = learning_rate

    model_args = ClassificationArgs()
    model_args.learning_rate = best_params
    model_args.num_train_epochs = 4
    model_args.use_multiprocessing = False
    model_args.overwrite_output_dir = True
    cardiff_bert = ClassificationModel(model_type="roberta", model_name="roberta-base", args=model_args, use_cuda=False)
    cardiff_bert.train_model(cardiff_data_bert_train)

    predicted, raw = cardiff_bert.predict(cardiff_data_bert_test_text)
    print(metrics.f1_score(cardiff_data_bert_test.labels, predicted))
    print("")
    
    # Section 2 using 1000 posts from other subreddits
    #Gathering data for use with BERT classifier
    all_sub_data_bert = pd.read_json("all_sub_data_bert.json")
    all_sub_data_bert_traintune, all_sub_data_bert_test = train_test_split(all_sub_data_bert, test_size=0.2 ,random_state=42)
    all_sub_data_bert_train, all_sub_data_bert_tune = train_test_split(all_sub_data_bert_traintune, test_size=0.1 ,random_state=42)

    all_sub_data_bert_test.columns = ["text", "labels"]
    all_sub_data_bert_tune.columns = ["text", "labels"]
    all_sub_data_bert_train.columns = ["text", "labels"]

    all_sub_data_bert_tune_text = all_sub_data_bert_tune.text.values.tolist()
    all_sub_data_bert_test_text = all_sub_data_bert_test.text.values.tolist()
    
    # Section 2a, testing how well the cardiff classifier can classify reddit posts from other subreddits
    print("Section 1b: Cardiff trained classifier, all subreddit testing")
    print("BERT")
    predicted, raw = cardiff_bert.predict(all_sub_data_bert_test_text)
    print(metrics.f1_score(all_sub_data_bert_test.labels, predicted))
    print("")

    # Section 2b, Testing how well a classifier trained on 1000 random reddit posts can classify narrative posts
    print("Section 2a: All subreddit trained classifier, All Subreddit testing")
    print("BERT")
    best_score = 0
    best_params = ""
    learning_rates = [2e-5, 3e-5, 4e-5, 5e-5]
    for learning_rate in learning_rates:
        model_args = ClassificationArgs()
        model_args.learning_rate = learning_rate
        model_args.num_train_epochs = 4
        model_args.use_multiprocessing = False
        model_args.overwrite_output_dir = True
        tuner = ClassificationModel(model_type="roberta", model_name="roberta-base", args=model_args, use_cuda=False)
        tuner.train_model(all_sub_data_bert_train)
        predicted, raw = tuner.predict(all_sub_data_bert_tune_text)
        
        f1_score = metrics.f1_score(all_sub_data_bert_tune.labels, predicted)

        if f1_score > best_score:
            best_score = f1_score
            best_params = learning_rate

    model_args = ClassificationArgs()
    model_args.learning_rate = best_params
    model_args.num_train_epochs = 4
    model_args.use_multiprocessing = False
    model_args.overwrite_output_dir = True
    all_sub_bert = ClassificationModel(model_type="roberta", model_name="roberta-base", args=model_args, use_cuda=False)
    all_sub_bert.train_model(all_sub_data_bert_train)

    predicted, raw = all_sub_bert.predict(all_sub_data_bert_test_text)
    print(metrics.f1_score(all_sub_data_bert_test.labels, predicted))
    print("")

    # Section 2c, testing how well a classifier trained on 1000 random reddit posts can classify cardiff reddit posts
    print("Section 2b: All subreddit trained classifier, Cardiff data testing")
    print("BERT")
    predicted, raw = all_sub_bert.predict(cardiff_data_bert_test_text)
    print(metrics.f1_score(cardiff_data_bert_test.labels, predicted))
    print("")

if __name__ == '__main__':
    run()

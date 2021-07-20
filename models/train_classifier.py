import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import logging
from argparse import ArgumentParser
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, f1_score, make_scorer
import pickle
from functools import partial
from time import time


def nltk_download():
    """
    Downloads required NLTK data.
    """
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')


def load_data(database_filepath, table_name='messages'):
    """
    Loads database with table containing messages dataset.

    Returns a dataframe containing the message data.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, engine)
    return df


def tokenize(text, remove_stop_words=False):
    """
    Custom tokenizer for messages.

    TODO: remove_stop_words - Boolean to control whether 
    """
    # case normalization (i.e. all lower case)
    text = text.lower()
    # punctuation removal
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # tokenize the text
    tokens = word_tokenize(text)
    # TODO
    # option stop words removal
    # if remove_stop_words:
    #     tokens = [
    #         token for token in tokens if token not in stopwords.words('english')
    #     ]

    # wordnet lemmatize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(token) for token in tokens]

    return text


def build_model(tokenizer=tokenize):
    """
    Builds a pipeline with grid search and cross validation
    for classifying messages.

    Adjust steps of the model by changing this function.

    Parameters:
        tokenizer -- Custom tokenizer function for messages.

    Returns:
        model ------ Scikit learn estimator.
    """

    # Pipeline with term frequence-inverse document frequency
    # vectorizer and a multioutput classifier wrapped classifier.
    pipeline = Pipeline([
        ('vect_tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ], verbose=True)

    # Params for grid search.
    parameters = {'clf__estimator__n_estimators': [1]}

    # Wrap the f1_score function with several params, mostly to
    # suppress zero division warnings.
    # https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions
    # Weighted average gets metrics weighted by support.
    # Zero division = 0 sets metric to 0 when there is no support.
    scorer = make_scorer(f1_score, average='weighted', zero_division=0)

    # Grid search.
    # Use cross validation = 5 for actual model training, 2 for testing.
    # n_jobs > 1 for take advantage of multi-core.
    # However, avoid n_jobs = -1, as using all processors may freeze
    # your workstation if working locally.
    model = GridSearchCV(pipeline, param_grid=parameters,
                         n_jobs=2, scoring=scorer, refit=True, cv=2, verbose=3)

    return model


def evaluate_model(model, X_test, Y_test):
    """
    Predict with a model.

    Params:
        model  - A trained sklearn model.
        X_test - Input data.
        Y_test - True results.
    Returns:
        scores - Pandas df with precision, recall, and f1 score for each output category.
    """
    # Predict with the model
    Y_preds = model.predict(X_test)
    # Evaluate precision, recall, and f1 score
    scores = classification_report(
        Y_test, Y_preds, output_dict=True, target_names=Y_test.columns, zero_division=0)
    # Return a dataframe of the scores
    return pd.DataFrame.from_dict(scores, orient='index')


def save_model(model, model_filepath):
    """
    Pickles and saves a classifier model.

    Params:
        model ----------- A sklearn model.
        model_filepath -- Filepath for the pickle file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def save_scores(scores, scores_filepath):
    """
    Saves scores dataframe to csv.

    Params:
        scores ----------- A pandas dataframe.
        scores_filepath -- Filepath for the scores csv.
    """
    scores.to_csv(scores_filepath)


def main():
    # Configure logging
    log_filepath = 'train_classifier.log'
    logging.basicConfig(filename=log_filepath,
                        level=logging.INFO,
                        format='%(asctime)s -- %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p'
                        )

    # Sys args parsing setup
    description = "Machine learning pipeline for training a classifier on message data."
    parser = ArgumentParser(description=description)
    parser.add_argument('messages_db',
                        help='Messages data in a sqlite database, e.g. ../data/disaster_response.db')
    parser.add_argument('filepath_prefix',
                        help='Output location prefix for final trained model and its scoring info.')
    # Get sys args
    args = parser.parse_args()

    # Check/download NLTK data for text processing.
    nltk_download()

    # Load the data
    df = load_data(args.messages_db)
    logging.info('Loading data successful.')

    # Split data into training and test
    Y = df.iloc[:, 4:]
    X = df['message']
    test_size = 0.2
    random_state = 7
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    logging.info(
        f'Split data with test size {test_size} and random state {random_state}')

    # Build a grid search cross validation model
    model = build_model(tokenizer=tokenize)
    logging.info('Model built with parameters:')
    logging.info('-----')
    logging.info(model.get_params())
    logging.info('-----')

    # Train the model.
    t_start = time()
    model.fit(X_train, Y_train)
    t_end = time()
    logging.info(f'Model trained in {(t_end - t_start)/60} minutes.')

    # Test the model
    scores = evaluate_model(model, X_test, Y_test)
    print(scores)

    # Save the model
    save_model(model, args.filepath_prefix + '.pkl')
    # Save the scores
    save_scores(scores, args.filepath_prefix + '.csv')


if __name__ == '__main__':
    main()

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
    # case normalization (i.e. all lower case)
    text = text.lower()
    # punctuation removal
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # tokenize the text
    tokens = word_tokenize(text)
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
    pipeline = Pipeline([
        ('vect_tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ], verbose=True)

    parameters = {'clf__estimator__n_estimators': [10]}

    # Wrap the f1_score function with several params.
    # https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions
    # Weighted average gets metrics weighted by support.
    # Zero division = 0 sets metric to 0 when there is no support.
    # f1_weighted = partial(f1_score, average='weighted', zero_division=0)
    # def scorer(y_true, y_pred):
    #     return f1_score(y_true, y_pred, average='weighted', zero_division=0)

    scorer = make_scorer(f1_score, average='weighted', zero_division=0)

    model = GridSearchCV(pipeline, param_grid=parameters,
                         n_jobs=2, scoring=scorer, refit=True, cv=2, verbose=3)

    return model


def evaluate_model(model, X_test, Y_test):
    # Predict with the model
    Y_preds = model.predict(X_test)
    # Evaluate precision, recall, and f1 score
    scores = classification_report(
        Y_test, Y_preds, output_dict=True, target_names=Y_test.columns, zero_division=0)
    # Return a dataframe of the scores
    return pd.DataFrame.from_dict(scores, orient='index')


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    # Configure logging
    logging.basicConfig(filename='train_classifier.log',
                        level=logging.INFO,
                        format='%(asctime)s -- %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p'
                        )

    # Sys args parsing setup
    description = "Machine learning pipeline for training a classifier on message data."
    parser = ArgumentParser(description=description)
    parser.add_argument('messages_db',
                        help='Messages data in a sqlite database, e.g. ../data/disaster_response.db')
    parser.add_argument('model_filepath',
                        help='Output location for final trained model.')
    # Get sys args
    args = parser.parse_args()

    # Check/download NLTK data for text processing.
    nltk_download()

    # Load the data
    df = load_data(args.messages_db)

    # Split data into training and test
    Y = df.iloc[:, 4:]
    X = df['message']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=7)

    # Build a grid search cross validation model
    model = build_model(tokenizer=tokenize)

    # Train the model
    model.fit(X_train, Y_train)

    # Test the model
    scores = evaluate_model(model, X_test, Y_test)
    print(scores)

    # Save the model
    save_model(model, args.model_filepath)

    ###
    # if len(sys.argv) == 3:
    #     database_filepath, model_filepath = sys.argv[1:]
    #     print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    #     X, Y, category_names = load_data(database_filepath)
    #     X_train, X_test, Y_train, Y_test = train_test_split(
    #         X, Y, test_size=0.2)

    #     print('Building model...')
    #     model = build_model()

    #     print('Training model...')
    #     model.fit(X_train, Y_train)

    #     print('Evaluating model...')
    #     evaluate_model(model, X_test, Y_test, category_names)

    #     print('Saving model...\n    MODEL: {}'.format(model_filepath))
    #     save_model(model, model_filepath)

    #     print('Trained model saved!')

    # else:
    #     print('Please provide the filepath of the disaster messages database '
    #           'as the first argument and the filepath of the pickle file to '
    #           'save the model to as the second argument. \n\nExample: python '
    #           'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

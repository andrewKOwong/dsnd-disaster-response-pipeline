import json
import plotly
import pandas as pd
import plotly.express as px

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

# TODO remove this and think about where/how to pickle the classifier
# so it doesn't necessarily call this... or harmonize
# the tokenizer functions.


def tokenize(text, remove_stop_words=False, stopwords=stopwords.words('english')):
    """
    Custom tokenizer for messages.

    Params:
        text --------------- Text to be tokenized.
        remove_stop_words -- Boolean to remove stop words or not.
        stopwords ---------- English stopwords by default.

    Returns:
        tokens ------------- Tokenized version of the input text.
    """
    # case normalization (i.e. all lower case)
    text = text.lower()
    # punctuation removal
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # tokenize the text
    tokens = word_tokenize(text)
    # Optional stop words removal.
    # Note: calling stopwords.words('english') here directly causes
    # pickling functions for some reason (which prevents multicore
    # processing).
    if remove_stop_words:
        tokens = [
            token for token in tokens if token not in stopwords
        ]

    # wordnet lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def tokenize_keep_stopwords(text):
    return tokenize(text, remove_stop_words=False)


def tokenize_remove_stopwords(text):
    return tokenize(text, remove_stop_words=True)


# Load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('messages', engine)

# Load model
model = joblib.load("../models/classifier.pkl")

# Load scores for plotting
score_df = pd.read_csv("../models/classifier.csv")
score_df = score_df.rename(columns={'Unnamed: 0': 'category'})

# Create figures in Plotly Express then serialize them to json strings.
# Bar plot of performance by category
perf_fig = px.bar(score_df, x='category', y=['precision', 'recall', 'f1-score'],
                  labels={'variable': 'Score Type', 'value': 'score'},
                  color_discrete_sequence=['gold', 'silver', '#c96'],
                  barmode='group',
                  )
perf_fig.update_layout(title_text='Classifier Performance by Category',
                       bargap=.35)
perf_fig.update_yaxes(range=[0, 1.1])
perf_fig = perf_fig.to_json()
# Scatterplot of F1 score vs support
# global_stat are global score variables
global_stat = ['micro avg', 'macro avg', 'weighted avg', 'samples avg']
f1_support_fig = px.scatter(score_df.query('category not in @global_stat'),
                            x='support',
                            y='f1-score',
                            text='category'
                            )
f1_support_fig.update_traces(textposition='top center')
f1_support_fig.update_layout(title_text='Classifier F1 Score vs. Support')
f1_support_fig.update_yaxes(range=[-0.025, 1.05])
f1_support_fig = f1_support_fig.to_json()
# Plot of the categories of the training data.
sums = df.iloc[:, 4:].sum()
sums_df = pd.concat([sums, df.shape[0] - sums], axis=1)
sums_df.columns = ['sum', 'remainder']
sums_fig = px.bar(sums_df, y=['sum', 'remainder'],
                  color_discrete_sequence=['blue', 'lightgray'])
sums_fig.update_layout(showlegend=False)
sums_fig = sums_fig.to_json()
# Plot of the proportion of messages in each category
# that are translated from their original language
proportion_translated = []
for cat in sums_df.index:
    # Proportion of message in a category that needed
    # translated from the original
    prop = 1 - df[df[cat] == 1].original.isna().mean()
    proportion_translated.append([cat, prop])
proportion_translated = pd.DataFrame(proportion_translated, columns=[
                                     'Category', 'Proportion Translated'])
translated_fig = px.bar(proportion_translated,
                        x='Category', y='Proportion Translated',)
translated_fig = translated_fig.to_json()

# Index displays graphs and info about model,
# and receives input text for classification.


@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html',
                           perf_fig=perf_fig,
                           f1_support_fig=f1_support_fig,
                           sums_fig=sums_fig,
                           translated_fig=translated_fig)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    print(df.columns)
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

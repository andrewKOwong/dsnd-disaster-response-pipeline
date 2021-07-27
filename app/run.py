import json
import plotly
import pandas as pd
import plotly.express as px

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens

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
df = pd.read_csv("../models/classifier.csv")
df = df.rename(columns={'Unnamed: 0': 'category'})

# Create figures in Plotly Express then serialize them to json strings
# Bar plot of performance by category
perf_fig = px.bar(df, x='category', y=['precision', 'recall', 'f1-score'],
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
f1_support_fig = px.scatter(df.query('category not in @global_stat'),
                            x='support',
                            y='f1-score',
                            text='category',
                            width=1000, height=1000
                            )
f1_support_fig.update_traces(textposition='top center')
f1_support_fig.update_layout(title_text='Classifier F1 Score vs. Support')
f1_support_fig.update_yaxes(range=[-0.025, 1.05])
f1_support_fig = f1_support_fig.to_json()


# Index displays graphs and info about model,
# and receives input text for classification.
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html', perf_fig=perf_fig, f1_support_fig=f1_support_fig)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
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

# Disaster Response Pipeline Project

## Project Summary

This project is part of the Udacity Data Scientist Nanodegree.

This goal of this project is to build an ETL (extract, transform, load) pipeline and a machine learning pipeline.

The dataset is composed of 26028 short messages that may or may not be relevant to disaster responders. Relevant messages are categorized as "related", and "related" messages are further subcategorized into 35 subcategories. 39% of all messages are translated into English from a non-English language (although a small number of these are in English regardless). Two plots describing the dataset are available in the web app (see below).

The English messages/translations were used to train a random forest model for prediction of whether messages are relevant and what subcategory of message they would belong to. Messages were first normalized to lower case, stripped of punctuation, stop words removed (I found this to be better than keeping stop words), and lemmatized with WordNet. 80% of the data set was then processed in an scikit-learn pipeline, first by TF-IDF (term frequency\*inverse document frequency) vectorization, then into a random forest model wrapped in a multi-output classifier. Scoring is by F1 score average weighted over subcategory size. I ended up using 100 estimators in the random forest.

The overall model F1 score is ~0.5 However, subcategory scores vary wildly, likely due to the subcategories being very unbalanced. A plot of F1 score vs support (size of the category) is shown in the web app.

The resulting classifier is available for prediction via a web app. The skeleton of this web app was provided by Udacity, and I wrote code to load in the model and display plots about the dataset and the model.

### Possible improvements:

The dataset is very unbalanced, leading to very poor scores among a number of the subcategories. As well, "related" messages probably make up too much of the data set, and the final model appears to predict false positives at a high rate. It may be worth dropping some of the smaller subcategories entirely. However, if it may be worth in high priority situations (such as imminent death) to use recall as a preferred scoring metric, as false negatives would be a worse outcome than false positives in such a scenario.

I did not experiment with different types of models. I suspect I could get the scores higher with more experimentation, but as the explicit goal of this project is to build a pipeline as opposed to optimizing the model per se, I elected not to do so for now.

## Instructions:

### Libraries required:

`numpy`  
`pandas`  
`sqlalchemy`  
`nltk`  
`scikit-learn` version >=0.24  
`plotly`  
`flask`

### Running the pipelines and the webapp:

1. Run the following commands in the project's root directory to set up your database and model.

   - Run the ETL pipeline that cleans data and stores it in a database:

     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

     You may also run `python data/process_data.py --help` if you want to change the filepaths.

     A log file `process_data.log` will also be generated.

   - Run the ML pipeline that trains a classifier and saves the model along with some extra info:

     `python models/train_classifier.py data/DisasterResponse.db models/classifier`

     This will save the model as a pickle file `classifier.pkl` as well as as a `classifier.csv` file with data about the model's performance for plotting in the web app.

     A log file `train_classifier.log` will also be generated.

     For more complete logging, you may also capture `stderr` and `stdoutput` by running in the shell:  
     `python train_classifier.py ../data/disaster_response.db classifier 2>&1 | tee stdout_stderr.log`

     This can be helpful for inspecting classifier training iterations.

     The pipeline is set train a random forest with stop word removal and 100 estimators. You may alter the grid search parameters inside the `build_model()` method.

2. To start the webapp, in the root directory run:  
   `python app/run.py`  
   The webapp will be available at: http://0.0.0.0:3001/

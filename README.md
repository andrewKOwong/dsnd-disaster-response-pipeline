# Disaster Response Pipeline Project

This goal of this project is to build an ETL (extract, transform, load) pipeline and a machine learning pipeline.

The dataset is composed of 26028 short messages that may or may not be relevant to disaster responders. Relevant messages are categorized as "related", and "related" messages are further subcategorized into 35 subcategories. 39% of all messages are translated into English from a non-English language (although a small number of these are in English regardless).

This project is part of the Udacity Data Scientist Nanodegree.

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python run.py`

3. Go to http://0.0.0.0:3001/

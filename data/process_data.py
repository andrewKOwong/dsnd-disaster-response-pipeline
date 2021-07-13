import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
from argparse import ArgumentParser


def load_data(messages_filepath, categories_filepath):
    """
    Loads csv files contains messages and categories, then merges them into one dataframe.

    Merging is by inner join. Some messages/categories will be duplicate entries.
    Also, some messages will have more than one corresponding non-duplicated
    category entry.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    merge_col = 'id'
    df = pd.merge(messages, categories, on=merge_col)
    logging.info(
        f'Messages and categories combined on {merge_col}. Dims: {df.shape}.')
    return df


def clean_data(df):
    """
    Transform combined messages+categories dataframe.

    Steps:
    1) Drop duplicates
    2) Extract categories and create new columns
    """
    # Drop duplicates
    initial_dim = df.shape
    df = df.drop_duplicates()
    dropped_dim = df.shape
    logging.info(f'Dropped {initial_dim[0] - dropped_dim[0]} duplicates.')

    # Extract categories
    # Make a categories df with each category in a column
    categories = df.categories.str.split(';', expand=True)
    # Grab the category column names by looking at the first row.
    # i.e. cells are in "cat_name-number" format, so we'll split the 'cat-name' out.
    row = categories.iloc[0, :]
    category_colnames = row.str.split('-').str[0]
    categories.columns = category_colnames
    # Now similarly split the text by column to grab the `number` part of the string
    for column in categories:
        # Getting the `number` part, and convert it to numeric type
        categories[column] = pd.to_numeric(
            categories[column].str.split('-').str[-1])

    # All categories columns should be 'int64'
    try:
        assert (categories.dtypes == 'int64').all()
    except:
        logging.error('Could not convert categories to "int64" data type.')
        sys.exit(1)

    # Replace categories column with new categories columns
    df = df.drop(labels=['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)

    # A small number of samples have the "related" category as
    # 2 (instead of 0 or 1).
    # These rows all are encoded as 0 for every other category.
    # This is probably some sort of bad data, so drop it.
    initial_dim = df.shape
    df = df[~(df['related'] == 2)]
    dropped_dim = df.shape
    logging.info(
        f"Dropped {initial_dim[0] - dropped_dim[0]} samples where 'related' == 2.")

    # Return the cleaned dataframe
    logging.info(f'Cleaned dataframe created with dims: {df.shape}')
    return df


def save_data(df, database_filename='messages.db', table_name='messages'):
    """
    Saves a cleaned dataframe of messages into an sqlite db.

    df                - cleaned messages dataframe.
    database_filename - name for the .db file
    table_name        - name for the table.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, index=False)
    logging.info(f'Saved to {database_filename}. Table: {table_name}.')


def main():
    # Configure logging
    logging.basicConfig(filename='process_data.log',
                        level=logging.INFO,
                        format='%(asctime)s -- %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %I:%M:%S %p'
                        )

    # Sys args parsing setup
    description = "Merge messages + categories csvs, clean data, and save into a database."
    parser = ArgumentParser(description=description)
    parser.add_argument('messages_filepath',
                        help='messages data as a csv, e.g. messages.csv')
    parser.add_argument('categories_filepath',
                        help='categories data as a csv, e.g. categories.csv')
    parser.add_argument(
        'database_filepath', help='path to save the output database, e.g. disaster_response.db')
    # Get sys args
    args = parser.parse_args()

    # Load the data
    print(('Loading data...\n'
           f'  MESSAGES: {args.messages_filepath}\n'
           f'  CATEGORIES: {args.categories_filepath}'))
    df = load_data(args.messages_filepath, args.categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)

    print(('Saving data...\n'
           f'  DATABASE: {args.database_filepath}'))
    save_data(df, args.database_filepath)

    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()

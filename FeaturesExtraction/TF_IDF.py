from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import jinja2

import pickle
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from Training.RandomForest import randomForest


def calculate(data, labels):
    vectorizer = TfidfVectorizer()
    # X, y = load_diabetes(return_X_y=True, as_frame=True)
    # X.head()

    # produce tfidf values
    X = vectorizer.fit_transform(data)

    # These are our words/tokens to be used as column names
    feature_names = vectorizer.get_feature_names_out()

    # I used the 5 sentences to index the table produced
    corpus_index = [sentence for sentence in data]

    df = pd.DataFrame(X.todense(), index=corpus_index, columns=feature_names)
    df.style

    # df.to_csv("Input/result", sep='\t', encoding='utf-8')

    # print(len(X)," ",len(labels))
    # print(X)

    randomForest(X, labels)
    # print(df)

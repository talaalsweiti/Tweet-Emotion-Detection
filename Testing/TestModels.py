import pickle

import pandas as pd
from sklearn import metrics

from Preprocessing.CleanData import clean
from Result import display_results


def test(tfidf, classifier_model, classifier):
    testing_tweets_file = pd.read_csv('Testing/TestingTweets.tsv', sep='\t')
    X_test = []
    y = []
    for i in range(len(testing_tweets_file)):
        y.append(testing_tweets_file.iloc[i][0])
        X_test.append(clean(testing_tweets_file.iloc[i][1]))

    X_test = tfidf.transform(X_test)

    # Use the model to make predictions on new data
    y_pred = classifier_model.predict(X_test)
    display_results(y, y_pred, X_test, classifier_model,classifier)

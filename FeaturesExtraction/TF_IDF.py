import csv

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def extract_features(tweets):
    tfidf = TfidfVectorizer()
    tfidf.fit(tweets)

    # Transform the training data into a matrix of TF-IDF scores
    X_train_tfidf = tfidf.transform(tweets)


    # These are our words/tokens to be used as column names
    feature_names = tfidf.get_feature_names_out()
    tweet_index = [tweet for tweet in tweets]

    df = pd.DataFrame(X_train_tfidf.todense(), index=tweet_index, columns=feature_names)
    df.style

    return X_train_tfidf

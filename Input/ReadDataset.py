import csv

import pandas as pd

from Preprocessing.CleanData import clean


def read_given_data():
    tweets = {}
    cleaned_tweets = []
    labels = []
    negative_tweets_file = pd.read_csv('Input/Negative+Tweets.tsv', sep='\t')
    positive_tweets_file = pd.read_csv('Input/Positive+Tweets.tsv', sep='\t')

    for i in range(len(negative_tweets_file)):
        tweets[clean(negative_tweets_file.iloc[i][0])] = "neg"

    for i in range(len(positive_tweets_file)):
        tweets[clean(positive_tweets_file.iloc[i][0])] = "pos"

    with open('cleanedData.csv', 'w', encoding="utf-8", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["sentiment", "tweet"])
        for tweet in tweets:
            if len(tweet) > 0:
                cleaned_tweets.append(tweet)
                labels.append(tweets[tweet])
                writer.writerow([tweets[tweet], tweet])
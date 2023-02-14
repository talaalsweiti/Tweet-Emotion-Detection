import csv

import pandas as pd

from FeaturesExtraction.TF_IDF import calculate
from Preprocessing.CleanData import clean
from Training.RandomForest import random_forest

if __name__ == '__main__':
    cleaned_tweets = []
    tweets = {}
    labels = []

    # negative_tweets_file = pd.read_csv('Input/Negative+Tweets.tsv', sep='\t')
    # positive_tweets_file = pd.read_csv('Input/Positive+Tweets.tsv', sep='\t')
    #
    # for i in range(len(negative_tweets_file)):
    #     tweets[clean(negative_tweets_file.iloc[i][0])] = "neg"
    #
    # for i in range(len(positive_tweets_file)):
    #     tweets[clean(positive_tweets_file.iloc[i][0])] = "pos"
    #
    #
    # with open('cleanedData.csv', 'w', encoding="utf-8", newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(["sentiment", "tweet"])
    #     for tweet in tweets:
    #         if len(tweet) > 0:
    #             cleaned_tweets.append(tweet)
    #             # print(tweet)
    #             labels.append(tweets[tweet])
    #             writer.writerow([tweets[tweet], tweet])

    positive_tweets_file = pd.read_csv('cleanedData.csv', sep=',')
    for i in range(len(positive_tweets_file)):
        labels.append(positive_tweets_file.iloc[i][0])
        cleaned_tweets.append(positive_tweets_file.iloc[i][1])

    X = calculate(cleaned_tweets)

    # print("Calculating finished")

    random_forest(X, labels)

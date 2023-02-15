import csv

import pandas as pd
from sklearn.model_selection import train_test_split

from FeaturesExtraction.TF_IDF import extract_features
from Preprocessing.CleanData import clean
from Training.NeuralNetwork import classical_neural_network
from Training.RandomForest import classical_random_forest, five_folds_random_forest


def read_given_data():
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
                # print(tweet)
                labels.append(tweets[tweet])
                writer.writerow([tweets[tweet], tweet])


if __name__ == '__main__':
    cleaned_tweets = []
    tweets = {}
    labels = []

    cleaned_tweets_file = pd.read_csv('cleanedData.csv', sep=',')
    for i in range(len(cleaned_tweets_file)):
        labels.append(cleaned_tweets_file.iloc[i][0])
        cleaned_tweets.append(cleaned_tweets_file.iloc[i][1])

    X = extract_features(cleaned_tweets)

    # classical_neural_network(X,labels)
    # classical_random_forest(X, labels)
    # five_folds_random_forest(X, labels)

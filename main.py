import pandas as pd

from FeaturesExtraction.TF_IDF import extract_features
from Training.RandomForest import five_folds_random_forest

if __name__ == '__main__':
    cleaned_tweets = []
    labels = []

    cleaned_tweets_file = pd.read_csv('cleanedData.csv', sep=',')
    for i in range(len(cleaned_tweets_file)):
        labels.append(cleaned_tweets_file.iloc[i][0])
        cleaned_tweets.append(cleaned_tweets_file.iloc[i][1])

    X = extract_features(cleaned_tweets)

    # classical_naive_bayes(X, labels)

    # classical_random_forest(X, labels)
    five_folds_random_forest(X, labels)

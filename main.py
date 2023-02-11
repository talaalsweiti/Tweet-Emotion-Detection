import pandas as pd

from FeaturesExtraction.TF_IDF import calculate
from Preprocessing.CleanData import clean
from Training.RandomForest import randomForest

if __name__ == '__main__':

    negativeTweetsFile = pd.read_csv('Input/Negative+Tweets.tsv', sep='\t')
    positiveTweetsFile = pd.read_csv('Input/Positive+Tweets.tsv', sep='\t')

    tweets = {}
    labels = []
    print("Cleaning started")

    for i in range(len(negativeTweetsFile)):
        tweets[clean(negativeTweetsFile.iloc[i][0])] = "neg"

    for i in range(len(positiveTweetsFile)):
        tweets[clean(positiveTweetsFile.iloc[i][0])] = "pos"

    print("Cleaning finished")

    cleanedTweets = []
    for tweet in tweets:
        cleanedTweets.append(tweet)
        labels.append(tweets[tweet])

    X = calculate(cleanedTweets)

    print("Calculating finished")

    randomForest(X, labels)

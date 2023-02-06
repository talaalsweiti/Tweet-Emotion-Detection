import pandas as pd

from Preprocessing.CleanData import clean

if __name__ == '__main__':

    negativeTweetsFile = pd.read_csv('Input/Negative+Tweets.tsv', sep='\t')
    negativeTweets = []

    for i in range(len(negativeTweetsFile)):
        tweet = negativeTweetsFile.iloc[i][1]
        negativeTweets.append(tweet)

    for i in range(5):
        print(negativeTweetsFile.iloc[i][1])
        print(clean(negativeTweetsFile.iloc[i][1]))
        print()

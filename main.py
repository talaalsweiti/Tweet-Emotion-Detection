import pandas as pd

from FeaturesExtraction.TF_IDF import calculate
from Preprocessing.CleanData import clean
import pandas as pd
import pickle
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

if __name__ == '__main__':

    negativeTweetsFile = pd.read_csv('Input/Negative+Tweets.tsv', sep='\t')
    negativeTweets = []

    positiveTweetsFile = pd.read_csv('Input/Positive+Tweets.tsv', sep='\t')
    positiveTweets = []
    positiveLabels = []
    #
    # for i in range(len(negativeTweetsFile)):
    #     negativeTweets.append(clean(negativeTweetsFile.iloc[i][1]))
    #
    # for i in range(5):
    #     print(negativeTweetsFile.iloc[i][1])
    #     print(negativeTweets[i])
    #     print()

    for i in range(len(positiveTweetsFile)):
        positiveTweets.append(clean(positiveTweetsFile.iloc[i][0]))
        # print(positiveTweetsFile.iloc[i][0])
        positiveLabels.append("pos")

    # for i in range(1):
    #     print(positiveTweetsFile.iloc[i][1])
    #     print(positiveTweets[i])
    #     print()
    # print(len(positiveTweetsFile))
    calculate(positiveTweets, positiveLabels)

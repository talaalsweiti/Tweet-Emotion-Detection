from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def extract_features(tweets):
    vectorizer = TfidfVectorizer()

    # produce tfidf values
    X = vectorizer.fit_transform(tweets)

    # These are our words/tokens to be used as column names
    feature_names = vectorizer.get_feature_names_out()

    tweet_index = [sentence for sentence in tweets]

    df = pd.DataFrame(X.todense(), index=tweet_index, columns=feature_names)
    df.style

    return X

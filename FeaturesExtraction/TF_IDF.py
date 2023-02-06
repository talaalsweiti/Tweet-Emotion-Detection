from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

import jinja2

def calculate(data):

    vectorizer = TfidfVectorizer()

    # produce tfidf values
    X = vectorizer.fit_transform(data)

    # These are our words/tokens to be used as column names
    feature_names = vectorizer.get_feature_names_out()

    # I used the 5 sentences to index the table produced
    corpus_index = [sentence for sentence in data]

    df = pd.DataFrame(X.todense(), index=corpus_index, columns=feature_names)
    df.style
    df.to_csv("Input/result", sep='\t', encoding='utf-8')

    # print(df)

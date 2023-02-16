import pickle

from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from FeaturesExtraction.TF_IDF import extract_features
from Result import display_results

nb_classifier = MultinomialNB()


def five_folds_naive_bayes(X, y,tweets):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(nb_classifier, X, y, cv=5)
    print("Naive Bayes")
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    for i, (train_index, test_index) in enumerate(kf.split(tweets)):
        print(f"Fold {i}")

        y_train = []
        X_train = []
        for index in train_index:
            X_train.append(tweets[index])
            y_train.append(y[index])
        X_train,tfidf = extract_features(X_train)
        y_test = []
        X_test = []
        for index in test_index:
            X_test.append(tweets[index])
            y_test.append(y[index])
        X_test = tfidf.transform(X_test)
        classifier = nb_classifier.fit(X_train, y_train)
        y_pred = nb_classifier.predict(X_test)
        display_results(y_test, y_pred, X_test, classifier, "Naive Bayes")
        print("------------------------------")


def classical_naive_bayes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Train the classifier on the training data
    classifier = nb_classifier.fit(X_train, y_train)

    # Use the classifier to make predictions on the testing data
    y_pred = nb_classifier.predict(X_test)

    with open('classical_naive_bayes.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    display_results(y_test, y_pred, X_test, classifier, "Naive Bayes")

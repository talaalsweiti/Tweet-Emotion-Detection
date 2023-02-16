import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from FeaturesExtraction.TF_IDF import extract_features
from Result import display_results

RForest_clf = RandomForestClassifier(n_estimators=100)


def five_folds_random_forest(X, y,tweets):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(RForest_clf, X, y, cv=kf)

    print("Random Forest")
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
        classifier = RForest_clf.fit(X_train, y_train)
        y_pred = RForest_clf.predict(X_test)
        display_results(y_test, y_pred, X_test, classifier, "Random Forest")
        print("------------------------------")



def classical_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    classifier = RForest_clf.fit(X_train, y_train)
    y_pred = RForest_clf.predict(X_test)
    # Save the trained model to a file using pickle
    # with open('classical_random_forest.pkl', 'wb') as f:
    #     pickle.dump(classifier, f)
    display_results(y_test, y_pred, X_test, classifier, "Random Forest")

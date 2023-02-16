import pickle

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_predict, cross_val_score

from FeaturesExtraction.TF_IDF import extract_features
from Result import display_results

clf = svm.SVC(kernel='linear')


def five_folds_svm(X, y, tweets):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5)
    print("SVM")
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    for i, (train_index, test_index) in enumerate(kf.split(tweets)):
        print(f"Fold {i}")
        y_train = []
        X_train = []
        for index in train_index:
            X_train.append(tweets[index])
            y_train.append(y[index])
        X_train, tfidf = extract_features(X_train)
        y_test = []
        X_test = []
        for index in test_index:
            X_test.append(tweets[index])
            y_test.append(y[index])
        X_test = tfidf.transform(X_test)
        classifier = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        display_results(y_test, y_pred, X_test, classifier, "SVM")
        print("------------------------------")


def classical_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Train the classifier on the training set
    classifier = clf.fit(X_train, y_train)

    # Test the classifier on the testing set and print the accuracy score
    y_pred = clf.predict(X_test)

    # with open('classical_svm.pkl', 'wb') as f:
    #     pickle.dump(classifier, f)

    display_results(y_test, y_pred, X_test, classifier, "SVM")

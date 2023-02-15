from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from Result import display_results

nb_classifier = MultinomialNB()


def five_folds_naive_bayes(X, y):
    # Create a KFold object with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(nb_classifier, X, y, cv=kf)
    # Compute accuracy, precision, and recall
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred, average='weighted')
    recall = metrics.recall_score(y, y_pred, average='weighted')

    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))


def classical_naive_bayes(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Train the classifier on the training data
    classifier = nb_classifier.fit(X_train, y_train)

    # Use the classifier to make predictions on the testing data
    y_pred = nb_classifier.predict(X_test)

    display_results(y_test, y_pred, X_test, classifier, "Naive Bayes")

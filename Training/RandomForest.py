import pickle

import numpy as np
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from Result import display_results

RForest_clf = RandomForestClassifier(n_estimators=100)


def five_folds_random_forest(X, y):
    # Create a KFold object with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(RForest_clf, X, y, cv=kf)
    # Compute accuracy, precision, and recall
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred, average='weighted')
    recall = metrics.recall_score(y, y_pred, average='weighted')

    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))


def classical_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    classifier = RForest_clf.fit(X_train, y_train)
    y_pred = RForest_clf.predict(X_test)
    display_results(y_test, y_pred, X_test, classifier, "Random Forest")

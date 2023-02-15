from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_predict

from Result import display_results

clf = svm.SVC(kernel='linear')

def five_folds_random_forest(X, y):

    # Create a KFold object with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(clf, X, y, cv=kf)
    # Compute accuracy, precision, and recall
    accuracy = metrics.accuracy_score(y, y_pred)
    precision = metrics.precision_score(y, y_pred, average='weighted')
    recall = metrics.recall_score(y, y_pred, average='weighted')

    print("Accuracy: {:.2f}".format(accuracy))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
def classical_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Train the classifier on the training set
    classifier = clf.fit(X_train, y_train)

    # Test the classifier on the testing set and print the accuracy score
    y_pred = clf.predict(X_test)

    display_results(y_test, y_pred, X_test, classifier, "SVM")

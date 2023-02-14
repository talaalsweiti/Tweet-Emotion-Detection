import pickle

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def five_foldes_random_forest(X, y):
    kf = KFold(n_splits=5,shuffle=True)

    # this is where splitting is done, i is the number of folds
    # train_index holds the indices of the training tweets (4/5 of the data)
    # test_index holds the indices of the testing tweets (1/5 of the data)
    for i, (train_index, test_index) in enumerate(kf.split(X)):

        print(f"\nFold {i} training data:\n")
        # for index in train_index - 1:
        #     print(X[train_index[index]], "-", y[train_index[index]])

        print(f"\nFold {i} testing data:\n")
        print(X[test_index[0]].data, "-", y[test_index[0]])
        print("------------------------------")


def classical_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # n_estimators is the number of trees in the forest
    RForest_clf = RandomForestClassifier(n_estimators=25)

    # train the model using 25% of the dataset
    classifier = RForest_clf.fit(X_train, y_train)

    # test the model using the remaining 75% of the dataset
    y_pred = RForest_clf.predict(X_test)

    # find the accuracy
    print('\n'"Accuracy of our Random Forest Classifier is: ",
          metrics.accuracy_score(y_test, y_pred) * 100, "\n")

    print('\n'"Precision score of our Random Forest Classifier is: ",
          metrics.precision_score(y_test, y_pred, pos_label='pos') * 100, "\n")

    print('\n'"Recall score of our Random Forest Classifier is: ",
          metrics.recall_score(y_test, y_pred, pos_label='pos') * 100, "\n")

    print('\n'"F1 score of our Random Forest Classifier is: ",
          metrics.f1_score(y_test, y_pred, pos_label='pos') * 100, "\n")

    class_names = ["pos", "neg"]

    # plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]

    pickle.dump(RForest_clf, open('model_25.pk1', 'wb'))

    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            classifier,
            X_test,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
        print()

    plt.show()

import pickle

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold



def five_folds_random_forest(X, y):
    rf = RandomForestClassifier()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X, y, cv=kf, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    y_pred = cross_val_predict(rf, X, y, cv=kf)

    accuracy = [accuracy_score(y[kf.test_fold == i], y_pred[kf.test_fold == i]) for i in range(5)]
    print("Accuracy for each fold:", accuracy)

    # Create a plot of the predicted values and true target values for each fold
    fig, ax = plt.subplots()
    ax.plot(y, 'ro', label='True Values')
    ax.plot(y_pred, 'bx', label='Predicted Values')
    ax.legend()
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Class Label')
    ax.set_title('Random Forest Predictions - 5-fold Cross Validation')
    plt.show()


def classical_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # n_estimators is the number of trees in the forest
    RForest_clf = RandomForestClassifier(n_estimators=25)
    # train the model using 75% of the dataset
    classifier = RForest_clf.fit(X_train, y_train)
    # test the model using the remaining 25% of the dataset
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

    # pickle.dump(RForest_clf, open('model_25.pk1', 'wb'))

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

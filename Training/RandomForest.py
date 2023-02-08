import numpy as np
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


def randomForest(X, y):
    # refer to the figure above to better understand this line
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # n_estimators is the number of trees in the forest, this can be
    # a parameter that you can play with to fine tune your model
    RForest_clf = RandomForestClassifier(n_estimators=100)

    # train the model using 25% of the dataset
    classifier = RForest_clf.fit(X_train, y_train)

    # test the model using the remaining 75% of the dataset
    y_pred = RForest_clf.predict(X_test)

    # find the accuracy
    print('\n'"Accuracy of our Random Forest Classifier is: ",
          metrics.accuracy_score(y_test, y_pred) * 100, "\n")

    class_names = ["pos", "neg"]

    # plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
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

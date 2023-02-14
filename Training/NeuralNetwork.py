from sklearn.neural_network import MLPClassifier

import pickle

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def classical_neural_network(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    NN_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                           hidden_layer_sizes=(3, 2), random_state=1)

    # train the model using 25% of the dataset
    classifier = NN_clf.fit(X_train, y_train)

    # test the model using the remaining 75% of the dataset
    y_pred = NN_clf.predict(X_test)

    # find the accuracy
    print('\n'"Accuracy of our Neural Netowrk Classifier is: ",
          metrics.accuracy_score(y_test, y_pred) * 100, "\n")

    class_names = ["pos", "neg"]

    # plot non-normalized confusion matrix
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]

    pickle.dump(NN_clf, open('model_25.pk1', 'wb'))

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

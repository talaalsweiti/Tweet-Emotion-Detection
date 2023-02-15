from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def display_results(y_test, y_pred, X_test, classifier, model):
    # find the accuracy
    print(f"Accuracy of our {model} Classifier is:{metrics.accuracy_score(y_test, y_pred) * 100} \n")

    print(classification_report(y_test, y_pred))

    class_names = ["pos", "neg"]

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

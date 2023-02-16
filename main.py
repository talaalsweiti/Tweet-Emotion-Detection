import csv
import pickle
import sys

import pandas as pd

from FeaturesExtraction.TF_IDF import extract_features
from Testing.TestModels import test
from Training.NaïveBayes import classical_naive_bayes, five_folds_naive_bayes
from Training.RandomForest import five_folds_random_forest, classical_random_forest
from Training.SVM import classical_svm, five_folds_svm

csv.field_size_limit(2147483647)
cleaned_tweets = []
labels = []
X = []


def display_mini_menu():
    print("Press 1 if you want to test unseen data")
    print("Else to return to main menu")



def display_menu():
    print("Choose an option: ")
    print("1- Show Random Forest results using classical method")
    print("2- Show Random Forest results using 5-folds cross validation")
    print("3- Show Naive Bayes results using classical method")
    print("4- Show Naive Bayes results using 5-folds cross validation")
    print("5- Show SVM results using classical method")
    print("6- Show SVM results using 5-folds cross validation")
    print("Else - Exit")


def store_models():
    classical_random_forest(X, labels)
    classical_naive_bayes(X, labels)
    classical_svm(X, labels)
    five_folds_random_forest(X, labels)
    five_folds_naive_bayes(X, labels)
    five_folds_svm(X, labels)


if __name__ == '__main__':

    cleaned_tweets_file = pd.read_csv('cleanedData.csv', sep=',')
    for i in range(len(cleaned_tweets_file)):
        labels.append(cleaned_tweets_file.iloc[i][0])
        cleaned_tweets.append(cleaned_tweets_file.iloc[i][1])

    X, stored_tfidf = extract_features(cleaned_tweets)

    # #Save the vectorizer
    # with open('tfidf.pickle', 'wb') as f:
    #     pickle.dump(stored_tfidf, f)

    with open('Models/tfidf.pickle', 'rb') as f:
        tfidf = pickle.load(f)

    print("Tweet Emotion Detection")

    while True:
        display_menu()
        val = input()
        if val == "1":
            classical_random_forest(X, labels)
            display_mini_menu()
            val2 = input()
            if val2 == "1":
                with open('Models/classical_random_forest.pkl', 'rb') as file:
                    classifier_model = pickle.load(file)
                test(tfidf, classifier_model, "Classical Random Forest")

        elif val == "2":
            five_folds_random_forest(X, labels, cleaned_tweets)

        elif val == "3":

            classical_naive_bayes(X,labels)
            display_mini_menu()
            val2 = input()
            if val2 == "1":
                with open('Models/classical_naive_bayes.pkl', 'rb') as file:
                    classifier_model = pickle.load(file)
                test(tfidf, classifier_model, "Classical Naive Bayes")

        elif val == "4":
            five_folds_naive_bayes(X, labels, cleaned_tweets)

        elif val == "5":
            classical_svm(X,labels)
            display_mini_menu()
            val2 = input()
            if val2 == "1":
                with open('Models/classical_svm.pkl', 'rb') as file:
                    classifier_model = pickle.load(file)
                test(tfidf, classifier_model, "Classical SVM")

        elif val == "6":
            five_folds_svm(X, labels, cleaned_tweets)

        else:
            sys.exit("Thank you for using or program!")

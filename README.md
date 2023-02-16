# Tweet-Emotion-Detection

This project has been done as a requirement for the Artifical Intelligence (ENCS3340) course at Birzeit University 
---
## Why I did this?
The aim of this project is to develop an automated solution that predicts the emotion expressed in a tweet based on extracted features from the tweet content. We consider Arabic text collected from Twitter, which can provide information having utility in a variety of ways, especially opining mining. This project proposes a method to classify text into two different categories: negative and positive. The followed approach is based on Machine Learning classification algorithms; Random Forest, Naïve Bayes, and Support Vector Machine.  


---

## Problem Formalization 
Input: A dataset of tweets, each labeled with a sentiment class (positive and negative).

Output: A model that can predict the sentiment of new, unseen tweets.

Objective: Develop a machine learning model that can accurately classify the sentiment of a tweet, with a focus on high accuracy, precision, recall, and F1-score.

Approach: This problem can be approached using traditional machine learning algorithms such as Naive Bayes, SVM, and Random Forest. The approach should include data preprocessing and feature extraction, model training and evaluation.



---
## General Strategy for Sentiment Analysis
### Data Preprocessing 
The given Tweets dataset requires removing all superfluous data, which includes special symbols, blanks spaces, stop words, numbers, repeating characters that occur more than twice, and Arabic diacritics. Emoji’s have been converted to words, and finally, Lemmatization has been applied on the tweets (convert any word to its base root mode).        

### Features Extraction
After the data pre-processing step, the next essential step is the choice of features on a refined dataset. Supervised machine learning classifiers require textual data in vector form to get trained on it. The textual features are converted into vector form using TF-IDF techniques. What actually Term Frequency(TF) means that, according to what often the term arises within the document? It’s measured by TF. This will be achievable with the intention of a term would seem a lot further in lengthy documents than short documents because every document is variant in extent. Meanwhile, invers document frequency (IDF) looks at how common (or uncommon) a word is amongst the corpus. 

To summarize the key intuition motivating TF-IDF is the importance of a term is inversely related to its frequency across documents.TF gives us information on how often a term appears in a document and IDF gives us information about the relative rarity of a term in the collection of documents. By multiplying these values together, we can get our final TF-IDF value.




### Proposed Models  

#### Decision Tree (Random Forest)
RF is a tree-based classifier that uses a random tree-generation input vector. In order to build numerous decision trees and a forest, RF uses random features. Then class labels of test data are predicted by aggregating voting of all trees. Higher weights are assigned to the decision trees with low value error. Overall prediction accuracy is improved by considering trees with low error rate. 

#### Naïve Bayes
In machine learning, Naive Bayes classifiers are a gathering of essential "probabilistic classifiers" based on Bayes' theorem that assumes independence among the features.  
In this project, Multinomial Naive Bayes has been used. It’s a variant of the Naive Bayes algorithm that is specifically designed for text classification problems. It assumes that the input features are counts of words or other discrete features, and models the conditional probability of the class given the frequency of the words in the document. 


#### Support Vector Machine (SVM)
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space (N is the number of features) that distinctly classifies the data points. 
The SVM kernel is a function that converts non separable problem to separable problem. It does some extremely complex data transformations then finds out the process to separate the data based on the labels or outputs defined. 



The evaluation for the performance for each classification model was based on the following:
1.	Accuracy: Accuracy measures the proportion of correctly classified samples to the total number of samples in the dataset. It is a general metric that indicates the overall correctness of the model's predictions.
2.	Precision: Precision measures the proportion of true positives (correctly predicted positive samples) to the total number of predicted positive samples. It is a metric that indicates how many of the positive predictions were actually correct.
3.	Recall: Recall measures the proportion of true positives to the total number of actual positive samples. It is a metric that indicates how many of the actual positive samples were correctly predicted.
4.	F1 score: F1 score is the harmonic mean of precision and recall. It is a metric that balances the importance of precision and recall in the model's performance.
5.	Confusion matrix: A confusion matrix is a table that shows the number of true positives, true negatives, false positives, and false negatives for a classification model. It provides a detailed breakdown of the model's performance and can be used to calculate other metrics such as precision, recall, and F1 score


---
## Languages And Tools:

- <img align="left" alt="Python" width="40px" src="https://www.pngfind.com/pngs/m/62-626208_python-logo-png-transparent-background-python-logo-png.png" /> <img align="left" alt="scikit learning" width="40px" src="https://github.com/scikit-learn/scikit-learn/blob/main/doc/logos/1280px-scikit-learn-logo.png?raw=true" /> <img align="left" alt="GitHub" width="50px" src="https://raw.githubusercontent.com/github/explore/78df643247d429f6cc873026c0622819ad797942/topics/github/github.png" />
<br/>

---




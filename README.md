# Tweet-Emotion-Detection

This project has been done as a requirement for the Artifical Intelligence (ENCS3340) course at Birzeit University 
---
## Why I did this?
The aim of this project is to develop an automated solution that predicts the emotion expressed in a tweet based on extracted features from the tweet content. We consider Arabic text collected from Twitter, which can provide information having utility in a variety of ways, especially opining mining. This project proposes a method to classify text into two different categories: negative and positive. The followed approach is based on Machine Learning classification algorithms; Random Forest, Naïve Bayes, and Support Vector Machine.  


---

## What is this project?



---
## General Strategy for Sentiment Analysis
### Data Preprocessing 
The given Tweets dataset requires removing all superfluous data, which includes special symbols, blanks spaces, stop words, numbers, repeating characters that occur more than twice, and Arabic diacritics. Emoji’s have been converted to words, and finally, Lemmatization has been applied on the tweets (convert any word to its base root mode).        

### Features Extraction
After the data pre-processing step, the next essential step is the choice of features on a refined dataset. Supervised machine learning classifiers require textual data in vector form to get trained on it. The textual features are converted into vector form using TF-IDF techniques. What actually Term Frequency(TF) means that, according to what often the term arises within the document? It’s measured by TF. This will be achievable with the intention of a term would seem a lot further in lengthy documents than short documents because every document is variant in extent. Meanwhile, invers document frequency (IDF) looks at how common (or uncommon) a word is amongst the corpus. 

To summarize the key intuition motivating TF-IDF is the importance of a term is inversely related to its frequency across documents.TF gives us information on how often a term appears in a document and IDF gives us information about the relative rarity of a term in the collection of documents. By multiplying these values together, we can get our final TF-IDF value.

### Proposed Models  

#### Decision Tree (Random Forest)
RF is a tree based classifier in which input vector generated trees randomly. RF uses random features, to create multiple decision trees, to make a forest. Then class labels of test data are predicted by aggregating voting of all trees. Higher weights are assigned to the decision trees with low value error. Overall prediction accuracy is improved by considering trees with low error rate.

#### Naïve Bayes


#### 4.3.3.	Support Vector Machine (SVM)
---
## Languages And Tools:

- <img align="left" alt="scikit learning" width="40px" src="https://github.com/scikit-learn/scikit-learn/blob/main/doc/logos/1280px-scikit-learn-logo.png?raw=true" />  <img align="left" alt="GitHub" width="50px" src="https://raw.githubusercontent.com/github/explore/78df643247d429f6cc873026c0622819ad797942/topics/github/github.png" />  
<br/>

---




import re

import pyarabic.araby as araby  # for removing diacritics
import qalsadi.lemmatizer

from nltk.tokenize import sent_tokenize, word_tokenize
from emot.core import emot
from nltk.stem.isri import ISRIStemmer
from emot.emo_unicode import UNICODE_EMOJI


def clean(sentence):
    def remove_diacritics(string):
        return araby.strip_diacritics(string)

    def convert_emojis(text):
        for emot in UNICODE_EMOJI:
            text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()))
        return text

    re_general_pattern = r"https?:\/\/.*[\r\n]*|#\w+|@\w+|\.{2,}"
    re_repeating_character_pattern = r"(\w)\1{2,}"
    lemmatizer = qalsadi.lemmatizer.Lemmatizer()
    st = ISRIStemmer()

    stopWordsFile = open("Preprocessing/stopwords.txt", "r", encoding='utf-8')
    stopWordsList = stopWordsFile.readlines()
    stopWords = []
    for word in stopWordsList:
        stopWords.append(word.replace("\n", ""))

    # 1- Removing URLs, Hashtags, Mentions, and repeating dots
    sentence = re.sub(re_general_pattern, "", sentence)

    # 2- Removing repeating characters that occur more than twice
    sentence = re.sub(re_repeating_character_pattern, r"\1", sentence)

    # 3- Removing arabic diacritics
    sentence = remove_diacritics(sentence)

    # 4-extract emojis
    emot_obj = emot()
    emojis = emot.emoji(emot_obj, sentence)

    # 5- Tokenization and punctuation removal (only alphanumeric)
    sequence = [token.lower() for token in word_tokenize(sentence) if token.isalpha()]

    # 7- Lemmatization
    sequence = [lemmatizer.lemmatize(token) for token in sequence]

    cleanedTweetList = []

    # remove stop words
    for cleanWord in sequence:
        cleanWord = st.stem(cleanWord)
        if cleanWord not in stopWords:
            cleanedTweetList.append(cleanWord)

    # concatenate emojis
    for emoji in emojis['value']:
        cleanedTweetList.append(convert_emojis(emoji))

    cleanedTweet = ' '.join([str(elem) for elem in cleanedTweetList])

    # convert_emojis(cleanedTweet)

    return cleanedTweet

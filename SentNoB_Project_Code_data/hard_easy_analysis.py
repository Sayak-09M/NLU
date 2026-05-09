import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import re

def compute_word_frequencies(texts):
    all_words = []
    for text in texts:
        words = str(text).split()
        all_words.extend(words)
    return Counter(all_words)

def build_hard_easy_analysis():
    train = pd.read_csv("SentNoB Dataset/Train.csv").dropna()
    test = pd.read_csv("SentNoB Dataset/Test.csv").dropna()

    print("Computing Train Word Frequencies...")
    freqs = compute_word_frequencies(train['Data'])

    def get_hard_word_ratio(text):
        words = str(text).split()
        if len(words) == 0:
            return 0
        hard_count = sum(1 for w in words if freqs.get(w, 0) < 5)
        return hard_count / len(words)

    print("Scoring Test set complexity...")
    test['hard_ratio'] = test['Data'].apply(get_hard_word_ratio)

    threshold = test['hard_ratio'].median()
    print(f"Median Hard Word Ratio in Test Setup: {threshold:.2f}")

    test_hard = test[test['hard_ratio'] > threshold]
    test_easy = test[test['hard_ratio'] <= threshold]

    print(f"Divided Test: {len(test_easy)} Easy Sentences, {len(test_hard)} Hard Sentences")

    print("Training SVM with (1,2) N-Grams for evaluation...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_tfidf = vectorizer.fit_transform(train['Data'])

    svm_model = SVC(kernel='linear', class_weight='balanced')
    svm_model.fit(X_train_tfidf, train['Label'])

    X_easy_tfidf = vectorizer.transform(test_easy['Data'])
    y_pred_easy = svm_model.predict(X_easy_tfidf)
    acc_easy = accuracy_score(test_easy['Label'], y_pred_easy)

    X_hard_tfidf = vectorizer.transform(test_hard['Data'])
    y_pred_hard = svm_model.predict(X_hard_tfidf)
    acc_hard = accuracy_score(test_hard['Label'], y_pred_hard)

    print("\n========= HARD VS EASY WORDS ANALYSIS =========")
    print(f"Evaluating Model: SVM (1,2 Grams)")
    print(f"Performance on EASY Texts (mostly common words): Accuracy = {acc_easy*100:.2f}%")
    print(f"Performance on HARD Texts (many rare/OOV words): Accuracy = {acc_hard*100:.2f}%")

    with open("hard_vs_easy_results.txt", "w") as f:
        f.write("========= HARD VS EASY WORDS ANALYSIS =========\n")
        f.write(f"Evaluating Model: SVM (1,2 Grams)\n")
        f.write(f"Performance on EASY Texts (mostly common words): Accuracy = {acc_easy*100:.2f}%\n")
        f.write(f"Performance on HARD Texts (many rare/OOV words): Accuracy = {acc_hard*100:.2f}%\n")

if __name__ == "__main__":
    build_hard_easy_analysis()

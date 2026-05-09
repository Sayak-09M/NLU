import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def plot_model_comparison():
    print("Loading datasets...")
    train = pd.read_csv("SentNoB Dataset/Train.csv").dropna()
    test = pd.read_csv("SentNoB Dataset/Test.csv").dropna()

    X_train, y_train = train['Data'], train['Label']

    all_words = []
    for text in X_train:
        all_words.extend(str(text).split())
    freqs = Counter(all_words)

    def get_hard_word_ratio(text):
        words = str(text).split()
        if len(words) == 0: return 0
        hard_count = sum(1 for w in words if freqs.get(w, 0) < 5)
        return hard_count / len(words)

    test['Hard_Word_Ratio'] = test['Data'].apply(get_hard_word_ratio)
    threshold = test['Hard_Word_Ratio'].median()

    test_hard = test[test['Hard_Word_Ratio'] > threshold].copy()
    test_easy = test[test['Hard_Word_Ratio'] <= threshold].copy()

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_easy_tfidf = vectorizer.transform(test_easy['Data'])
    X_hard_tfidf = vectorizer.transform(test_hard['Data'])

    models = {
        "Logistic Reg": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel='linear', class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": MultinomialNB()
    }

    results = []
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        acc_easy = accuracy_score(test_easy['Label'], model.predict(X_easy_tfidf)) * 100
        acc_hard = accuracy_score(test_hard['Label'], model.predict(X_hard_tfidf)) * 100

        results.append({"Model": name, "Accuracy": acc_easy, "Context Type": "Easy Contexts (Standard Bangla)"})
        results.append({"Model": name, "Accuracy": acc_hard, "Context Type": "Hard Contexts (Dialect/Noise)"})

    df_results = pd.DataFrame(results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    ax = sns.barplot(x="Model", y="Accuracy", hue="Context Type", data=df_results, palette=["#2ecc71", "#e74c3c"])

    plt.title('Algorithm Durability: Performance Drop-off on Noisy Bangla Texts', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Machine Learning Model', fontsize=12, fontweight='bold')
    plt.ylim(30, 75)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 10),
                    textcoords = 'offset points',
                    fontsize=10, fontweight='bold', color='black')

    plt.legend(title='Text Difficulty', title_fontsize='11', fontsize='10', loc='lower right')
    plt.tight_layout()
    plt.savefig("model_comparison_hard_easy.png", dpi=300)
    print("Saved -> model_comparison_hard_easy.png")

if __name__ == "__main__":
    plot_model_comparison()

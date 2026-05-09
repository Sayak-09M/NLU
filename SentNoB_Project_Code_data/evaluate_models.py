import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score

def load_sentnob():
    train = pd.read_csv("SentNoB Dataset/Train.csv").dropna()
    test = pd.read_csv("SentNoB Dataset/Test.csv").dropna()
    return train['Data'], train['Label'], test['Data'], test['Label']

def evaluate_ml():
    X_train, y_train, X_test, y_test = load_sentnob()
    print(f"Loaded SentNoB: Train={len(X_train)} Test={len(X_test)}")

    ngrams_to_test = {
        "1-Gram (Unigram)": (1, 1),
        "2-Gram (Bigram)": (2, 2),
        "3-Gram (Trigram)": (3, 3),
        "1-to-2 Grams (Uni+Bi)": (1, 2),
        "1-to-3 Grams (Uni+Bi+Tri)": (1, 3)
    }

    results = []

    for name, ngram_range in ngrams_to_test.items():
        print(f"\n--- Testing {name} {ngram_range} ---")
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=ngram_range)

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_tfidf, y_train)
        y_pred_lr = lr.predict(X_test_tfidf)
        lr_acc = accuracy_score(y_test, y_pred_lr)
        lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')

        svm_model = SVC(kernel='linear', class_weight='balanced')
        svm_model.fit(X_train_tfidf, y_train)
        y_pred_svm = svm_model.predict(X_test_tfidf)
        svm_acc = accuracy_score(y_test, y_pred_svm)
        svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')

        results.append({
            "N-Gram Setting": name,
            "LR Accuracy": lr_acc, "LR F1": lr_f1,
            "SVM Accuracy": svm_acc, "SVM F1": svm_f1
        })
        print(f"LR Acc: {lr_acc:.4f} | SVM Acc: {svm_acc:.4f}")

    df_res = pd.DataFrame(results)
    df_res.to_csv("ngram_comparison_results.csv", index=False)
    print("\n[SUCCESS] N-gram comparison completed. Results saved to ngram_comparison_results.csv")
    print(df_res.to_string())

if __name__ == "__main__":
    evaluate_ml()

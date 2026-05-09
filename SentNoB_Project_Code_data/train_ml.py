import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

def train_ml_models(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\n[ML] Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)

    lr_acc = accuracy_score(y_test, y_pred_lr)
    lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')

    print("Logistic Regression Results:")
    print(classification_report(y_test, y_pred_lr))

    print("\n[ML] Training SVM...")
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train_tfidf, y_train)
    y_pred_svm = svm_model.predict(X_test_tfidf)

    svm_acc = accuracy_score(y_test, y_pred_svm)
    svm_f1 = f1_score(y_test, y_pred_svm, average='weighted')

    print("SVM Results:")
    print(classification_report(y_test, y_pred_svm))

    results = [
        {"Model": "Logistic Regression", "Accuracy": lr_acc, "F1-Score": lr_f1},
        {"Model": "SVM", "Accuracy": svm_acc, "F1-Score": svm_f1}
    ]

    return results

if __name__ == "__main__":
    ml_results = train_ml_models("cleaned_data.csv")
    pd.DataFrame(ml_results).to_csv("ml_results.csv", index=False)
    print("\nML Results saved to ml_results.csv")

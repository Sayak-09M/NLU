import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.font_manager as fm

mpl.rcParams['font.family'] = 'Arial Unicode MS'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def run_advanced_analysis():
    print("Loading datasets...")
    train = pd.read_csv("SentNoB Dataset/Train.csv").dropna()
    test = pd.read_csv("SentNoB Dataset/Test.csv").dropna()

    X_train, y_train = train['Data'], train['Label']
    X_test, y_test = test['Data'], test['Label']

    print("Training Best Model (LR with Uni+Bi grams)...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_tfidf, y_train)
    y_pred = lr.predict(X_test_tfidf)

    report = classification_report(y_test, y_pred, target_names=["Neutral", "Positive", "Negative"])
    print("\n--- Detailed Classification Report ---")
    print(report)
    with open("classification_report_detailed.txt", "w") as f:
        f.write("--- Detailed Classification Report ---\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Neutral", "Positive", "Negative"], yticklabels=["Neutral", "Positive", "Negative"])
    plt.title('Confusion Matrix: Logistic Regression (1,2 Grams)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved -> confusion_matrix.png")

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = lr.coef_

    if coefs.shape[0] == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        classes = ["Neutral Insights", "Positive Insights", "Negative Insights"]
        colors = ['gray', 'green', 'red']

        prop = fm.FontProperties(fname='/Library/Fonts/Arial Unicode.ttf')

        for i in range(3):
            top10_idx = np.argsort(coefs[i])[-10:]
            top10_features = feature_names[top10_idx]
            top10_coefs = coefs[i][top10_idx]

            axes[i].barh(top10_features, top10_coefs, color=colors[i])
            axes[i].set_title(classes[i])
            axes[i].set_xlabel("Feature Coefficient Weight")

            axes[i].set_yticks(range(len(top10_features)))
            axes[i].set_yticklabels(top10_features, fontproperties=prop)

        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved -> feature_importance.png")

    test_analysis = pd.DataFrame({'Text': X_test, 'Actual': y_test, 'Predicted': y_pred})
    misclassified = test_analysis[test_analysis['Actual'] != test_analysis['Predicted']]

    misclassified.to_csv("misclassified_sentences.csv", index=False)
    print(f"Saved -> misclassified_sentences.csv ({len(misclassified)} errors found to analyze manually)")

    print("\n[SUCCESS] Advanced Analysis Pipeline Completed! Outputs saved.")

if __name__ == "__main__":
    run_advanced_analysis()

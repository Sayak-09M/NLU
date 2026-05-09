import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def run_all_with_bert():
    print("Loading datasets...")
    train = pd.read_csv("SentNoB Dataset/Train.csv").dropna()
    test = pd.read_csv("SentNoB Dataset/Test.csv").dropna()

    X_train, y_train = train['Data'], train['Label']

    print("Computing Hard/Easy Split...")
    all_words = []
    for text in X_train:
        all_words.extend(str(text).split())
    freqs = Counter(all_words)

    def get_hard_word_ratio(text):
        words = str(text).split()
        if len(words) == 0: return 0
        hard_count = sum(1 for w in words if freqs.get(w, 0) < 5)
        return hard_count / len(words)

    test['hard_ratio'] = test['Data'].apply(get_hard_word_ratio)
    threshold = test['hard_ratio'].median()

    test_hard = test[test['hard_ratio'] > threshold].copy()
    test_easy = test[test['hard_ratio'] <= threshold].copy()

    results = []

    print("Training ML Baselines (Word + Char Features)...")
    word_vect = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), analyzer='word')
    char_vect = TfidfVectorizer(max_features=5000, ngram_range=(2, 4), analyzer='char')

    vectorizer = FeatureUnion([
        ("word", word_vect),
        ("char", char_vect)
    ])

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_easy_tfidf = vectorizer.transform(test_easy['Data'])
    X_hard_tfidf = vectorizer.transform(test_hard['Data'])

    models = {
        "Logistic Reg (Word+Char)": LogisticRegression(max_iter=1000),
        "SVM (Word+Char)": SVC(kernel='linear', class_weight='balanced')
    }

    X_test_full_tfidf = vectorizer.transform(test['Data'])

    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        acc_global = accuracy_score(test['Label'], model.predict(X_test_full_tfidf)) * 100
        acc_easy = accuracy_score(test_easy['Label'], model.predict(X_easy_tfidf)) * 100
        acc_hard = accuracy_score(test_hard['Label'], model.predict(X_hard_tfidf)) * 100

        results.append({"Model": name, "Accuracy": acc_easy, "Context Type": "Easy Contexts (Standard Bangla)"})
        results.append({"Model": name, "Accuracy": acc_hard, "Context Type": "Hard Contexts (Dialect/Noise)"})
        print(f"{name} -> Global: {acc_global:.2f} | Easy: {acc_easy:.2f} | Hard: {acc_hard:.2f} | Drop (Global-Hard): {acc_global - acc_hard:.2f}")

    print("Tokenizing Data for BanglaBERT Finetuning...")
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
    def tokenize_func(examples):
        return tokenizer([str(t) for t in examples["Data"]], padding="max_length", truncation=True, max_length=128)

    train_ds = Dataset.from_pandas(train[['Data', 'Label']]).map(tokenize_func, batched=True).rename_column("Label", "labels")
    test_ds = Dataset.from_pandas(test[['Data', 'Label']]).map(tokenize_func, batched=True).rename_column("Label", "labels")
    easy_ds = Dataset.from_pandas(test_easy[['Data', 'Label']]).map(tokenize_func, batched=True).rename_column("Label", "labels")
    hard_ds = Dataset.from_pandas(test_hard[['Data', 'Label']]).map(tokenize_func, batched=True).rename_column("Label", "labels")

    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    easy_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    hard_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    print("Fine-tuning BanglaBERT (1 Fast Epoch)...")
    model = AutoModelForSequenceClassification.from_pretrained("csebuetnlp/banglabert", num_labels=3)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {'accuracy': accuracy_score(labels, predictions)}

    training_args = TrainingArguments(
        output_dir="./bert_results",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=500
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, compute_metrics=compute_metrics)
    trainer.train()

    print("Evaluating BanglaBERT on Global, Easy, and Hard Subsets...")
    eval_global = trainer.evaluate(eval_dataset=test_ds)['eval_accuracy'] * 100
    eval_easy = trainer.evaluate(eval_dataset=easy_ds)['eval_accuracy'] * 100
    eval_hard = trainer.evaluate(eval_dataset=hard_ds)['eval_accuracy'] * 100
    print(f"BanglaBERT -> Global: {eval_global:.2f} | Easy: {eval_easy:.2f} | Hard: {eval_hard:.2f} | Drop (Global-Hard): {eval_global - eval_hard:.2f}")

    results.append({"Model": "BanglaBERT (Pre-trained + FT)", "Accuracy": eval_easy, "Context Type": "Easy Contexts (Standard Bangla)"})
    results.append({"Model": "BanglaBERT (Pre-trained + FT)", "Accuracy": eval_hard, "Context Type": "Hard Contexts (Dialect/Noise)"})

    print("Generating Graphical Export...")
    df_results = pd.DataFrame(results)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="Accuracy", hue="Context Type", data=df_results, palette=["#2ecc71", "#e74c3c"])

    plt.title('Baseline ML vs. Deep Learning (BanglaBERT) Performance on Text Difficulty', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.ylim(30, 85)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha = 'center', va = 'center',
                    xytext = (0, 10),
                    textcoords = 'offset points',
                    fontsize=10, fontweight='bold')

    plt.legend(title='Semantic Complexity', loc='lower right')
    plt.tight_layout()
    plt.savefig("master_model_comparison_with_bert.png", dpi=300)
    print("Saved -> master_model_comparison_with_bert.png")

if __name__ == "__main__":
    run_all_with_bert()

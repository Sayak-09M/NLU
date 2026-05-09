import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def run_banglabert():
    print("Loading SentNoB Data...")
    train_df = pd.read_csv("SentNoB Dataset/Train.csv").dropna()
    test_df = pd.read_csv("SentNoB Dataset/Test.csv").dropna()

    train_dataset = Dataset.from_pandas(train_df[['Data', 'Label']])
    test_dataset = Dataset.from_pandas(test_df[['Data', 'Label']])

    model_name = "csebuetnlp/banglabert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        texts = [str(t) for t in examples["Data"]]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=128)

    print("Tokenizing data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    tokenized_train = tokenized_train.rename_column("Label", "labels")
    tokenized_test = tokenized_test.rename_column("Label", "labels")
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./transformer_results_sentnob",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=500
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("\nEvaluating on Test Set...")
    eval_results = trainer.evaluate(eval_dataset=tokenized_test)

    res = f"BanglaBERT Accuracy: {eval_results['eval_accuracy']:.4f}\nBanglaBERT F1: {eval_results['eval_f1']:.4f}"
    print(res)
    with open("banglabert_results.txt", "w") as f:
        f.write(res)

if __name__ == "__main__":
    run_banglabert()

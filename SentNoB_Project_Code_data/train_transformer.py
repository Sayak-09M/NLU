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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_transformer(csv_path):
    df = pd.read_csv(csv_path).dropna()

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df[['cleaned_text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['cleaned_text', 'label']])

    model_name = "csebuetnlp/banglabert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing data...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./transformer_results",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
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

    print("Evaluating...")
    eval_results = trainer.evaluate()

    return [
        {"Model": "BanglaBERT", "Accuracy": eval_results["eval_accuracy"], "F1-Score": eval_results["eval_f1"]}
    ]

if __name__ == "__main__":
    transformer_results = train_transformer("cleaned_data.csv")
    pd.DataFrame(transformer_results).to_csv("transformer_results.csv", index=False)
    print("\nTransformer results saved to transformer_results.csv")

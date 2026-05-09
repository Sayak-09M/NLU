import pandas as pd
import re
import string
from indicnlp.normalize.indic_normalize import BengaliNormalizer

def load_data(pos_path, neg_path):
    texts = []
    labels = []

    with open(pos_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                texts.append(t)
                labels.append(1)

    with open(neg_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                texts.append(t)
                labels.append(0)

    return pd.DataFrame({"text": texts, "label": labels})

def clean_bengali_text(text):
    normalizer = BengaliNormalizer()
    text = normalizer.normalize(text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    df = load_data("all_positive_8500.txt", "all_negative_3307.txt")
    print(f"Loaded {len(df)} rows.")
    df['cleaned_text'] = df['text'].apply(clean_bengali_text)
    df.to_csv("cleaned_data.csv", index=False)
    print("Saved cleaned data to cleaned_data.csv")

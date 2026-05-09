import pandas as pd
from collections import Counter

def export_hard_easy():
    print("Loading SentNoB Train and Test sets...")
    train = pd.read_csv("SentNoB Dataset/Train.csv").dropna()
    test = pd.read_csv("SentNoB Dataset/Test.csv").dropna()

    print("Computing Word Frequencies to extract Hard Words...")
    all_words = []
    for text in train['Data']:
        all_words.extend(str(text).split())
    freqs = Counter(all_words)

    hard_words = [word for word, count in freqs.items() if count < 5]
    easy_words = [word for word, count in freqs.items() if count >= 5]

    df_hard_vocab = pd.DataFrame({"Hard_Word": hard_words, "Frequency": [freqs[w] for w in hard_words]})
    df_hard_vocab = df_hard_vocab.sort_values(by="Frequency", ascending=False)
    df_hard_vocab.to_csv("Hard_Words_Vocabulary.csv", index=False)
    print(f"Saved -> Hard_Words_Vocabulary.csv ({len(hard_words)} words)")

    print("Scoring Sentences...")
    def get_hard_word_ratio(text):
        words = str(text).split()
        if len(words) == 0: return 0
        hard_count = sum(1 for w in words if freqs.get(w, 0) < 5)
        return hard_count / len(words)

    test['Hard_Word_Ratio'] = test['Data'].apply(get_hard_word_ratio)
    threshold = test['Hard_Word_Ratio'].median()

    test_hard = test[test['Hard_Word_Ratio'] > threshold].copy()
    test_easy = test[test['Hard_Word_Ratio'] <= threshold].copy()

    test_hard.to_csv("Hard_Sentences.csv", index=False)
    test_easy.to_csv("Easy_Sentences.csv", index=False)
    print(f"Saved -> Hard_Sentences.csv ({len(test_hard)} sentences)")
    print(f"Saved -> Easy_Sentences.csv ({len(test_easy)} sentences)")
    print("Done!")

if __name__ == "__main__":
    export_hard_easy()

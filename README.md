# NLU
SentNoB Sentiment Analysis Project
=====================================

This folder contains the complete, cleaned codebase for the SentNoB sentiment analysis project. The work here is a replication and extension of the research paper "SentNoB: A Dataset for Analysing Sentiment on Noisy Bangla Texts" by Islam et al. (EMNLP 2021). The goal was not just to run the baselines, but to actually understand why models fail — particularly on dialects, misspellings, and noisy online text written in Bangla.

The entire pipeline is written in Python, runs end-to-end without manual annotation, and produces visualizations, reports, and notebooks automatically.



The Dataset
------------


    dataset/Train.csv         The main training data (12,575 sentences)
    dataset/Test.csv          The evaluation set (1,586 sentences)
    dataset/Val.csv           A validation split included in the original dataset

    dataset/Easy_Sentences.csv         Test sentences scored as linguistically simpler
    dataset/Hard_Sentences.csv         Test sentences with high rates of rare/dialectal words
    dataset/Hard_Words_Vocabulary.csv  Every rare word extracted from the training corpus
    dataset/misclassified_sentences.csv  All sentences the model got wrong
    dataset/ngram_comparison_results.csv  Accuracy scores across different N-gram settings
    dataset/ml_results.csv             Final ML model results
    dataset/transformer_results.csv    BanglaBERT results
    dataset/cleaned_data.csv           The cleaned version of the raw training corpus


Python Scripts
---------------

Each script has a single, well-defined responsibility. You can run them individually or chain them together.

preprocess.py
    Takes the raw positive and negative sentence text files and turns them into a clean CSV.
    It normalizes Bengali Unicode, strips punctuation, and collapses extra whitespace.
    Run this first if you are starting from the raw txt files.

train_ml.py
    Trains Logistic Regression and SVM models on the cleaned_data.csv file.
    It uses TF-IDF with 1-2 gram features (which is the best setting we found).
    Saves results to ml_results.csv.

train_transformer.py
    Fine-tunes BanglaBERT on the cleaned dataset using the HuggingFace Trainer.
    Supports GPU, Apple MPS (for Mac), and CPU automatically.
    Saves results to transformer_results.csv.

evaluate_models.py
    Runs a systematic comparison of five different N-gram configurations across
    both Logistic Regression and SVM. This is how we figured out that 1-2 grams
    gives the best performance. Saves results to ngram_comparison_results.csv.

run_banglabert.py
    Runs BanglaBERT on the official SentNoB Train/Test split (not the custom cleaned data).
    Uses all three sentiment classes (Neutral, Positive, Negative).
    Saves accuracy and F1 scores to banglabert_results.txt.

run_and_plot_with_bert.py
    The main comprehensive evaluation script. Runs all models (ML baselines and BanglaBERT)
    on global, easy, and hard subsets. Generates the master comparison chart saved as
    master_model_comparison_with_bert.png.

advanced_analysis.py
    Produces detailed diagnostics for the best model: a confusion matrix, a feature
    importance chart showing which words most strongly drive each sentiment class,
    and a CSV of all misclassified sentences.

hard_easy_analysis.py
    Standalone script for the hard/easy split analysis using SVM.
    Outputs a text summary of accuracy on easy vs. hard test sets.

export_hard_easy.py
    Extracts and saves the hard word vocabulary and the hard/easy sentence splits to CSV.
    Useful for doing manual inspection of what the model is struggling with.

plot_model_comparison.py
    Generates a grouped bar chart showing how Logistic Regression, SVM,
    Random Forest, and Naive Bayes each perform on easy vs. hard texts.


test_font.py
    A small utility script that tests different system fonts to see which ones correctly
    render Bengali script in matplotlib plots. Not part of the main pipeline.


How to Run the Pipeline
------------------------

If you want to run everything from scratch, here is the logical order:

    1. python preprocess.py
       This creates cleaned_data.csv from the raw text files.

    2. python train_ml.py
       Trains the ML baseline models.

    3. python evaluate_models.py
       Compares N-gram configurations and saves a summary table.

    4. python export_hard_easy.py
       Creates the hard/easy sentence splits and the rare word vocabulary.

    5. python advanced_analysis.py
       Generates confusion matrix, feature importance chart, and misclassified sentences.

    6. python plot_model_comparison.py
       Generates the model comparison visualization across multiple algorithms.

    7. python run_banglabert.py
       Fine-tunes and evaluates BanglaBERT on the official SentNoB data.

    8. python run_and_plot_with_bert.py
       Runs the full combined evaluation across easy and hard subsets with all models.




Dependencies
-------------

To install the required packages, run:

    pip install pandas numpy scikit-learn matplotlib seaborn fpdf2 python-pptx nbformat
    pip install torch transformers datasets
    pip install indic-nlp-library

For BanglaBERT on Apple Silicon, PyTorch with MPS support is required.
For GPU training, install the appropriate CUDA version of PyTorch.


References
-----------

Islam, K. I., Islam, M. S., Kar, S., and Amin, M. R. (2021). SentNoB: A Dataset for
Analysing Sentiment on Noisy Bangla Texts. In Findings of ACL: EMNLP 2021, pp. 3265-3271.

Bhattacharjee, A., et al. (2022). BanglaBERT: Language Model Pretraining and Evaluating
on Low-Resource NLP Tasks. In Findings of ACL: NAACL 2022, pp. 1045-1060.

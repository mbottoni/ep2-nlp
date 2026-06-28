# Probing BERTimbau for Vowel Density

Can a Portuguese transformer's contextual embeddings recover a purely *surface*
property of the text? This project fine-tunes **BERTimbau**
(`neuralmind/bert-base-portuguese-cased`) on real product reviews from the
**B2W-Reviews01** dataset and trains it to predict each review's **vowel
density** — the fraction of letters that are vowels.

## The probe

Each review is reduced to a single number, its vowel density, and the model
predicts that number from the review text alone:

- **Regression** — predict the continuous vowel density, scored with RMSE and MAE.
- **Classification** — bucket density into three bands (low `< 1/3`, mid
  `1/3–2/3`, high `> 2/3`) and predict the band, in both an **unbalanced** and a
  **class-balanced** setting.

## Baselines

To check whether the embeddings actually carry the signal — rather than the task
being trivially easy — the fine-tuned model is compared against simple
heuristics: the global average density, the density of just the **first** word,
and the density of just the **last** word of each review.

## Data

[B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) — a corpus of
Brazilian e-commerce product reviews (title, rating, recommendation, and a
free-text review). Place `B2W-Reviews01.csv` under `data/`, or point the notebook
at your own copy.

## Running

The full pipeline lives in [`playground.ipynb`](playground.ipynb): preprocessing,
BERTimbau fine-tuning, evaluation, and the baseline comparisons. It was developed
on Google Colab (a GPU is recommended) — update the dataset path at the top of
the notebook before running.

## Stack

PyTorch · Hugging Face Transformers · BERTimbau · scikit-learn · pandas.

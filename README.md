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

## Install

```bash
pip install -e ".[dev]"     # editable install + pytest/ruff
# or just the runtime deps:
pip install -r requirements.txt
```

## Usage

The project installs a `bertimbau-probing` command:

```bash
bertimbau-probing baselines                  # naive density baselines (no GPU)
bertimbau-probing regression                 # fine-tune the regression probe
bertimbau-probing classification --balanced  # classifier on balanced classes
```

Common flags: `--data-path`, `--model-name`, `--samples`, `--epochs`,
`--batch-size`. Fine-tuning needs a GPU; the baselines run anywhere.

Equivalent `make` targets are provided: `make baselines`, `make regression`,
`make classification`, `make test`, `make lint`.

## Project layout

```
src/bertimbau_probing/
├── config.py      # hyperparameters and paths (Config dataclass)
├── vowels.py      # vowel-density target + density bands (pure Python)
├── data.py        # load / preprocess / target / split / balance (pandas)
├── baselines.py   # first-word / last-word / global density baselines
├── datasets.py    # tokenizing torch Dataset + DataLoaders
├── models.py      # BERTimbau regression and classification heads
├── train.py       # training and prediction loops
├── metrics.py     # RMSE / MAE / R² / Pearson, accuracy + confusion matrix
└── cli.py         # `bertimbau-probing` entry point
tests/             # unit tests for the torch-free modules
notebooks/         # original exploratory notebook
data/              # place B2W-Reviews01.csv here
```

The `vowels`, `data`, `baselines` and `metrics` modules have no deep-learning
dependencies, so they are unit-tested in CI without installing torch.

## Data

[B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01) — a corpus of
Brazilian e-commerce product reviews (title, rating, recommendation, and a
free-text review). Place `B2W-Reviews01.csv` under `data/`, or pass
`--data-path` to point at your own copy.

## Tests

```bash
pytest -q       # torch-free unit tests
ruff check src tests
```

## Stack

PyTorch · Hugging Face Transformers · BERTimbau · scikit-learn · pandas.

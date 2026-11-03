# Income bias: measuring and mitigating sex bias in an Adult-income classifier

This project follows the same architecture as [`ml/`](../ml) (the MNIST
digit-recognition example this repo also contains): a small PyTorch model,
a `train.py` that writes everything about a run to `runs/<id>/`, a
torch-free `runs.py` that reads those files back, and a Streamlit
dashboard (`app.py`) that launches training as a subprocess and polls the
run's files live. Here the task is tabular (predict income, not digits),
and the whole point of the project is to find and fix a bias rather than
just measure accuracy.

## The problem

The [UCI Adult / Census Income dataset](https://www.openml.org/d/1590)
(1994 US Census data, ~45k rows after cleaning) is one of the most
studied fairness benchmarks in machine learning. The task is: predict
whether a person's income is above $50,000/year from their age,
education, occupation, hours worked, and similar features.

The dataset also records each person's sex. We deliberately **exclude
sex from the model's input features** — the model never sees it — but we
keep it around as a *protected attribute*, used only to measure and
correct bias, never to predict with.

**The bias:** in the training data, women are a much smaller share of
the `>$50K` earners than they are of the population overall. A plain
classifier trained on this data, with no correction, picks up that
pattern: even among people who *actually* earn more than $50K, it
correctly flags far fewer of the women than the men. That gap in recall
(formally: equal opportunity, Hardt/Price/Srebro 2016) is the bias this
project targets — not overall accuracy, which can look fine even while
this gap is large, since women are a minority of the positive class to
begin with.

**The value we use the data to find:** the "TPR gap" — male recall minus
female recall, i.e. the difference in how often qualified men vs.
qualified women are correctly identified as high earners — both before
and after each mitigation, on a held-out test set the model never
trains or tunes on.

## The mitigations

Selected with `--bias-mitigation {none,oversample,reweight}` in
`train.py` (see `bias_mitigation.py` for the implementation and math):

- **`none`** — no correction. The baseline everything else is compared
  against.
- **`oversample`** — at the start of every training epoch, rows where
  `sex == female AND label == >50K` are duplicated at random until that
  group's share of the `>50K` training pool matches women's overall
  share of the training population. Nothing is removed; only the
  under-represented group's positive examples are topped up.
- **`reweight`** — [Kamiran & Calders (2012)](https://link.springer.com/article/10.1007/s10115-011-0463-8)
  reweighing. Every training example gets a fixed per-sample loss weight
  so that, under the weighted training distribution, sex and income
  become statistically independent. No rows are added or removed — some
  just count more or less in the loss.

## Architecture

| File | Role |
|---|---|
| `choices.py` | Plain string constants (e.g. model names) shared by both the training process and the dashboard, with zero imports — importing it never pulls in torch. |
| `census_data.py` | Downloads (via `sklearn.datasets.fetch_openml`), cleans, encodes, splits (70/15/15, seeded), and standardizes the Adult dataset; caches the result to `data/adult.npz`. |
| `bias_mitigation.py` | The bias this project addresses, and the oversampling / reweighing implementations described above. |
| `model.py` | Two small PyTorch models: `logreg` (one linear layer) and `mlp` (two hidden layers + dropout). |
| `fairness_metrics.py` | Pure numpy; no torch import. Computes per-group (male/female) accuracy, recall, precision, selection rate, and the headline `tpr_gap` and `selection_rate_gap`. |
| `train.py` | Trains one model/mitigation/seed combination, writing `config.json`, live `status.json`, `metrics.jsonl` (one line per epoch), the best checkpoint (`model.pt`, selected by validation accuracy), and a one-time test-set evaluation (`test.npz`, `test_report.json`) to `runs/<run-id>/`. |
| `runs.py` | No-torch reader for everything `train.py` writes — used by `app.py` so the dashboard works even while training dependencies are still installing. |
| `app.py` | Streamlit dashboard: start a run, watch it train live, inspect its test-set fairness report, and compare every finished run side by side. |

Train/val/test discipline, copied from `ml/`'s own README: the
checkpoint saved as "best" is chosen by **validation** accuracy only; the
**test** set is scored exactly once, at the very end of training, so the
reported test numbers stay an honest, unpeeked-at estimate.

## Running it

From the repo root, via pixi:

```bash
# Run the test suite (pure pandas/numpy/sklearn logic — no network, no GPU)
pixi run -e fairness-income test

# Train one model (downloads and caches the Adult dataset on first run)
pixi run -e fairness-income train -- --model mlp --bias-mitigation none --epochs 20
pixi run -e fairness-income train -- --model mlp --bias-mitigation oversample --epochs 20
pixi run -e fairness-income train -- --model mlp --bias-mitigation reweight --epochs 20

# Launch the dashboard (start more runs from the sidebar, or just browse
# the ones you already trained from the command line above)
pixi run -e fairness-income app
```

Or directly with Python, from inside `fairness_income/`:

```bash
python train.py --model mlp --bias-mitigation reweight --epochs 20
streamlit run app.py
```

`pixi run -e fairness-income lint` runs `ruff` over the project.

## Results

Train the same model/epoch/seed combination under all three
`--bias-mitigation` settings and compare their `runs/<id>/test_report.json`
(or just open the "Compare runs" tab in the dashboard) to fill in this
table:

| bias_mitigation | test accuracy | male recall | female recall | TPR gap |
|---|---|---|---|---|
| none       | — | — | — | — |
| oversample | — | — | — | — |
| reweight   | — | — | — | — |

A successful mitigation brings the TPR gap much closer to 0 while
costing only a small amount of overall test accuracy — the same
trade-off `ml/`'s own crossed-7s experiment demonstrates for its
mitigation.

## Screenshot

_Add a screenshot of the dashboard here (e.g. the "Compare runs" tab
after training all three settings) before submitting._

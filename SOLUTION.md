# SMILES-2026 Hallucination Detection Solution

## Reproducibility

Recommended environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONIOENCODING=utf-8
python solution.py
```

On Linux or Colab use `source .venv/bin/activate` instead of the Windows
activation command. On Linux/Colab the encoding line is usually not needed; if
it is needed, use `export PYTHONIOENCODING=utf-8`.

The script uses the original `solution.py` entry point. It loads
`Qwen/Qwen2.5-0.5B`, extracts hidden states for `data/dataset.csv` and
`data/test.csv`, trains the probe, writes `results.json`, and saves
`predictions.csv`.

## Final Approach

I modified the three allowed files:

- `aggregation.py`
- `probe.py`
- `splitting.py`

The aggregation step uses several late transformer layers instead of only the
last layer. For each selected layer it keeps the last token representation,
mean pooled tail representation, mean pooled full sequence representation, and
simple difference features. I also added a small number of statistics such as
activation norms, cosine similarities between layers, and normalized sequence
length. This keeps the method simple while giving the classifier more than one
view of the answer representation.

The probe is a regularized logistic regression ensemble. Features are
standardized, optionally reduced with PCA when the feature space is large, and
then several logistic regression models with different regularization strengths
are averaged. I used this instead of a larger neural network because the
training set is small, so a linear model with regularization is less likely to
overfit badly.

The split strategy is 5-fold stratified cross-validation. Inside each fold a
small validation subset is used for threshold tuning. This gives a more stable
estimate than a single random split while preserving the label ratio.

## Experiments and Notes

I considered using only the final token from the last transformer layer, but it
throws away information from earlier layers and from the answer span. I also
considered a small MLP probe, but for this dataset size it is easier to
overfit, and the results can be less stable across splits.

The final solution is intentionally not very complicated: most of the gain
should come from better hidden-state pooling and regularization rather than a
large classifier.

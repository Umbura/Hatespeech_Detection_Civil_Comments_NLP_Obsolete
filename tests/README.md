# Tests

The automated suite validates repository integrity, historical leakage protections, and the deterministic parts of the repaired hierarchical target strategy. It does not claim model-quality or training coverage.

Run the suite from the repository root with:

```bash
python -m pip install pandas numpy scikit-learn imbalanced-learn 'iterative-stratification>=0.1.9'
python -m unittest discover -s tests -p "test_*.py" -v
```

Current checks cover:

- expected repository layout;
- historical notebook JSON/nbformat integrity and preservation;
- README references to the current notebook path;
- train/validation isolation in the leakage-safe legacy comparison path;
- training-only resampling and tokenizer fitting in that legacy path;
- Stage 1 and Stage 2 target extraction from the original fractional scores;
- preservation of overlapping fine-grained labels;
- absence of a synthetic `non_toxic` output target;
- toxicity-gate coverage accounting;
- hierarchical fold disjointness and full validation coverage;
- iterative multilabel stratification of toxic samples;
- absence of Stage 2 oversampling in the hierarchical split strategy.

TensorFlow training, real-dataset gate analysis, model metrics, overfitting, and fairness are intentionally outside CI and require separate experimental validation.

# Tests

The automated suite validates repository integrity, historical leakage protections, and the deterministic parts of the repaired hierarchical target strategy. It does not claim model-quality or training coverage.

Run the suite from the repository root with the lightweight validation dependencies:

```bash
python -m pip install pandas==3.0.5 numpy==2.5.2 scikit-learn==1.9.0 imbalanced-learn==0.14.2 iterative-stratification==0.1.9
python -m pip check
python -m unittest discover -s tests -p "test_*.py" -v
```

Current checks cover:

- expected repository layout and the Python 3.12 runtime marker;
- historical notebook JSON/nbformat integrity and preservation;
- README references to the current notebook path;
- namespaced `google/civil_comments` runtime dataset identifiers;
- train/validation isolation in the leakage-safe legacy comparison path;
- training-only resampling and tokenizer fitting in that legacy path;
- Stage 1 and Stage 2 target extraction from the original fractional scores;
- preservation of overlapping fine-grained labels;
- absence of a synthetic `non_toxic` output target;
- the selected default `0.4` toxicity gate and explicit alternative-threshold behavior;
- toxicity-gate coverage accounting;
- hierarchical fold disjointness and full validation coverage;
- iterative multilabel stratification of toxic samples;
- absence of Stage 2 oversampling in the hierarchical split strategy.

TensorFlow training, replacement model metrics, overfitting repair, and fairness remain outside CI and require separate experimental validation. The real-dataset gate analysis has been executed separately and its measured evidence is recorded in `docs/TARGET_STRATEGY.md`.

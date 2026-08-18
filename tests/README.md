# Tests

The automated suite now covers both repository integrity and the isolation guarantees of the repaired cross-validation preparation path. It still does not claim end-to-end model-quality or training coverage.

Install the lightweight validation dependencies and run the suite from the repository root:

```bash
python -m pip install pandas numpy scikit-learn imbalanced-learn
python -m unittest discover -s tests -p "test_*.py" -v
```

Current checks cover:

- expected repository layout and canonical repaired-pipeline files;
- notebook JSON/nbformat integrity and preservation under `notebooks/`;
- README references to the historical notebook path;
- disjoint training/validation indices;
- training-only resampling with source-row traceability;
- validation rows remaining untouched by balancing;
- validation text never being used to fit the fold tokenizer;
- complete validation coverage across stratified folds.

The suite does not execute TensorFlow training, download Civil Comments, validate final target semantics, or prove that overfitting is resolved.

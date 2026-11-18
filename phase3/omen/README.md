# PathSnap Phase 3 · Omen Review 1

The canonical models use the Parquet inputs published at Phase 2 commit `59f573e2fdd76d546069bac0b204f1c64590e502`. The manifest fixes 64 training trips and 16 validation trips. All training and CV reads come from the existing Phase 2 artifacts. The final fits use the full 64-trip training partition. No student-test or evaluator data was used in this run.

## Measured results

| Model | Training-only 3-fold CV macro F1 | Validation accuracy | Precision | Recall | Macro F1 | Log loss | Conditional Top-1 | Coverage-aware Top-1 | Fit s | Predict s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7594 | 86.27% | 0.5192 | 0.8971 | 0.7859 | 0.2924 | 74.97% | **74.41%** | 4.98 | 0.038 |
| XGBoost | 0.8754 | 91.11% | 0.7084 | 0.6719 | 0.8189 | 0.2024 | 70.11% | 69.59% | 8.40 | 0.152 |

Logistic Regression ranks more reference arcs first. XGBoost achieves better candidate-row macro F1 and log loss. Use coverage-aware Top-1 for the classifier ranking comparison. Use separate sequence-level and continuity metrics when the team decoder is available. [Comparison chart](review1/comparison.png)

## Dataset audit

| Partition | Trips | GPS observations | Candidate rows | Positive rows | Observations with no candidates | Observations with candidates but no positive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 64 | 168,957 | 1,143,501 | 148,487 (12.99%) | 0 | 20,470 |
| Validation | 16 | 39,460 | 266,367 | 39,166 (14.70%) | 0 | 294 |

Parquet IDs match the manifest, with zero overlap. The audit found zero NaNs, infinities, duplicate arc candidates, or inconsistent labels within an observation. Observation counts are verified against the Phase 2 coverage ledger. Validation counts all 294 observations without a positive as misses in coverage-aware Top-1. Conditional Top-1 uses only the 39,166 observations with a positive candidate.

`is_correct` is a binary **proxy** label. Phase 2 searches the ordered reference route around an alignment pointer, then falls back to the closest candidate that appears anywhere in that route. A candidate-independent exact match per GPS point is unavailable in the source dataset. If no candidate is on the route, the whole observation group stays negative. The label generator adds metadata without changing candidate membership or rankings. The route-free test independently constructs the same 22 feature columns and candidate IDs for trip `00000005` without reading its `.route` file.

Candidate macro F1 is sklearn's `f1_score(..., average="macro", labels=[0, 1])` using a 0.5 threshold. Phase 2's baseline evaluator reports an observation-level precision/recall harmonic mean in its `macro_f1` field. Those historical values use a different definition. The Phase 2 baseline Top-1 denominator counts aligned observations. Compare it to conditional Top-1 when using the same trips and weighting.

## Features for the viva

The trainer imports the exact ordered `FeatureEngineer.FEATURE_COLUMNS` list. Distances and along-arc projection measure geometric fit. GPS speed, headings, cosine alignment, and counter-flow distinguish arcs at similar distances. Arc length and endpoint degrees describe network structure. Step GPS distance and shortest-path features from the previous GPS-only rank-1 candidate express whether the move is plausible on the directed graph. All 22 inputs are numeric. The trip and observation IDs, candidate and reference arc IDs, coordinates, and timestamps stay out of the model input.

Logistic Regression uses standardization, an L2-regularized logistic objective, and balanced class weights because only about 13% of training candidate rows are positive. Each fit learns its own scaler. Class weighting changes the training objective, so the resulting probabilities may need calibration before use as emissions. XGBoost uses the CPU histogram algorithm, 150 depth-5 trees, learning rate 0.08, and 0.85 row and column subsampling. Both models use seed 42 and sequential execution with at most four numerical worker threads.

## Reproduce from the repository root

Use the repository's Phase 2 dependencies plus XGBoost. The Review 1 environment used Python 3.14.6, XGBoost 3.4.1, scikit-learn 1.9.1, pandas 3.1.0, NumPy 2.5.1, PyArrow 25.0.1, joblib 1.6.0, and Matplotlib 3.11.1. Use the same versions when loading the bundled joblib artifacts.

```powershell
python -B phase3/omen/train_canonical.py --output review1-reproduction.zip
python -B -m unittest discover -s phase3/omen -p test_canonical.py -v
```

Use a fresh output path. The committed [experiment archive](review1/experiment.zip) contains the two fitted joblib models, native XGBoost weights, per-candidate validation probabilities with arc IDs, validation observation coverage flags, the audit, detailed metrics, and fold membership. The [audit](review1/audit.json), [Logistic metrics](review1/logistic.json), and [XGBoost metrics](review1/xgboost.json) can also be read directly.

For the six-model comparison, export the same field names as the per-model `team_summary.json` in the archive. This run keeps student-test and route-continuity fields null because it measured candidate classification on the canonical validation partition. Put the 16-trip results in the detailed `validation` object. The registry accepts a trusted archive per supported model name with `<name>/results.json` and `<name>/model.joblib`. It checks feature ordering and binary probability support before exposing the estimator.

## Earlier Omen work

The [preliminary viva notes](preliminary/OMEN_VIVA.md), original Omen scripts and FastAPI scaffold, compact model archives, and summaries are under `preliminary/`. The original experiment remains in the local Hackathon workspace. Its 20-feature grouped CV macro F1 was 0.9376 for Logistic Regression and 0.9573 for XGBoost. Its exploratory student-test directed Top-1 was 90.48% and 92.61%. Those runs used different candidate generation, proxy labels, features, and splits. The student-test set was inspected during feature experiments, so those numbers are exploratory.

## After Review 1

Compare all six models on the fixed Phase 2 split. Feed their saved probabilities into the existing Phase 2 graph-constrained HMM and measure directed Top-1 and sequence continuity. Use a training-only grouped protocol for Bayesian tuning after the comparison. The existing FastAPI scaffold is stored with the preliminary source for later deployment work.

## Folium verification

Opened the existing [Phase 1 map](../../phase1/results/plots/interactive_map_sample.html) and [Phase 2 map](../../phase2/results/plots/interactive_map_matching.html) on the Omen. Both rendered basemap tiles, vector overlays, and layer controls without browser errors. The Phase 3 audit also cross-checked all 80 raw track observation counts against the Phase 2 coverage ledger.

# PathSnap Omen accuracy audit

## What the audit established

The Phase 1 structural audit found the expected 80 training trips and preserved GPS anomalies. No Phase 1 corruption was demonstrated. Phase 2 has a **reference-measurement flaw**. Its labeler searches retrieved candidates for an arc on the supplied route. Its reported proxy coverage therefore depends on retrieval, and its anywhere-on-route fallback can bypass trip chronology. On the 16 validation trips, 4,072 labels could not be retained by an earliest-occurrence scan constrained to move forward along the route. This is a scan diagnostic, since one forward jump can affect many later labels.

The earlier 92.61% run used 80 fitted trips, 10 repeatedly inspected student-test trips, a different candidate pool, and GPS-window and bidirectional path features. Exact reversal errors, especially on trip `00000084`, dominated the old canonical XGBoost misses. This trip contributes 16,031 of 39,460 observations. Holding the old canonical XGBoost scores fixed and changing only the labels to the new proxy moved Top-1 from 69.59% to 69.43%. Refitting the same classifier settings against the new proxy yielded 71.01%. These controlled changes explain a small part of the historical 23-point difference.

## Frozen contract for further team runs

Use **`omen-phase2-route-first-v1`** for new controlled experiments. Keep the existing six-model historical leaderboard labeled with its Phase 2 candidate-window proxy. Publish new scores as a separate table.

- The original Phase 2 64/16 trip manifest stays fixed with seed 42. Only the 64 training trips enter fitting, model selection, and grouped CV.
- The original Phase 2 candidate pool and ordered 22 numerical feature columns stay fixed. The pool tries a 60 m KD-tree sample search, falls back to 120 m only when empty, and finally retrieves the closest sample's arc if necessary. It keeps up to ten candidates. It retrieved more independent references than the old 75 m/eight-candidate pool on both splits.
- The new proxy first aligns all observations to ordered reference-route geometry. Dynamic programming keeps nondecreasing route positions, including distinct occurrences of repeated arc IDs. It uses geometric distance and a bounded heading penalty for reliable GPS movement. Heading windows stop across gaps over 30 seconds. No candidate list enters this alignment.
- Candidate variables already computed by Phase 2 are unchanged. The exported keys and targets are `trip_id`, `obs_index`, `candidate_arc_id`, `true_arc_id`, and `is_correct`. The two arc IDs have different roles. `candidate_arc_id` identifies the retrieved candidate. `true_arc_id` is the scoring proxy, even when it falls outside the pool. Only the ordered 22 columns listed in the manifest enter the models.
- Coverage-aware directed Top-1 includes every GPS observation. Conditional Top-1 includes only observations where the independent proxy arc was retrieved. Select one arc per observation with ties broken by spatial rank and arc ID. Macro trip Top-1 gives each trip equal weight. Candidate macro F1 uses both classes at threshold 0.5.
- A reference proxy remains an estimate. The official evaluator's alignment and scoring formula are unknown. Use the same proxy across methods, and keep labels and route references outside inference inputs.

The frozen [manifest](results/dataset_manifest.json) includes trip IDs, source hashes, feature order, rules, and file hashes. The 64 training trips hold 168,957 observations and 1,143,501 candidates. The 16 validation trips hold 39,460 observations and 266,367 candidates. The gap-aware proxy retains 38,884 training and 7,827 validation observations flagged as geometrically near-tied. There were 105 and 28 GPS gaps over 30 seconds, respectively.

### Controlled retrieval and label comparison

All rows below use the same 39,460 validation observations and 16 trips. The model is the same rank-1 nearest heuristic throughout this table. The candidate pool and reference alignment are varied separately. Per-trip counts, mean and median candidates, proxy agreement, and opposite-direction discrepancies are in [candidate_audit.csv](results/candidate_audit.csv).

| Candidate pool | Reference alignment | Has candidates | Reference in pool | Nearest Top-1, all GPS |
| --- | --- | ---: | ---: | ---: |
| Omen 75 m, up to 8 | Candidate-window | 39,234 | 38,910 | 60.53% |
| Omen 75 m, up to 8 | Global geometric | 39,234 | 38,778 | 58.78% |
| Omen 75 m, up to 8 | Route-first, gap-aware | 39,234 | 38,772 | 59.07% |
| Phase 2, up to 10 | Candidate-window | 39,460 | 39,166 | 60.65% |
| Phase 2, up to 10 | Global geometric | 39,460 | 39,087 | 58.82% |
| Phase 2, up to 10 | Route-first, gap-aware | 39,460 | 39,079 | 57.77% |

Phase 2's independent retrieval ceiling is **99.03% on validation** and **87.48% on training**. The old pool retrieves 145,735 training references versus 147,799 for Phase 2. Training trips `00000054` and `00000033` alone account for 19,650 of the 21,158 independent misses. The ceiling measures how often rankers can select the proxy from the pool. On validation, the old geometry-only and gap-aware alignments agree at 38,076 observations and choose opposite directions at 862.

## CPU booster comparison

Every score below uses the frozen route-first labels. The model rows share the same 22 features, candidate pool and trip manifest. Each validation fold for model selection and tuning came from the 64 training trips.

| Model | Training CV micro Top-1 | Validation micro Top-1 | Conditional Top-1 | Validation trip-macro Top-1 | Worst trip (00000084) | Candidate macro F1 | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost baseline | 70.59% | **71.01%** | 71.70% | 81.96% | 51.67% | 0.8265 | 0.2010 |
| LightGBM baseline | 70.12% | 70.47% | 71.16% | 81.77% | 50.55% | 0.8216 | 0.2303 |
| XGBoost, Optuna TPE | **70.61%** | 70.93% | 71.62% | 82.02% | 51.36% | 0.8261 | 0.2012 |

Training CV selected XGBoost for the five-trial TPE search. The best trial used depth 5, 120 estimators, learning rate 0.1104, subsample 0.7060, column sample 0.8496, L2 0.3662, and minimum child weight 11.0263. The tuned fit gained 0.02 points in training CV and lost 0.08 points in validation micro Top-1. The baseline is the stronger observed micro result and is the artifact used for the API smoke test.

| Model | Fit | Validate probabilities | Exact reverse errors | Mean directed arc-set F1 | Mean continuity |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost baseline | 7.26 s | 0.108 s | 7,588 | 0.8085 | 74.83% |
| LightGBM baseline | 5.28 s | 0.398 s | 7,607 | 0.8077 | 73.92% |
| XGBoost, Optuna TPE | 6.70 s | 0.097 s | 7,608 | 0.8101 | 75.28% |

These route metrics use consecutive deduplication of classifier outputs before HMM or path bridging. They differ in stage from a reconstructed HMM route. The per-trip metrics, route sizes, and continuity accompany the [comparison](results/comparison.json). The [XGBoost baseline](results/xgboost_baseline.zip), [LightGBM baseline](results/lightgbm_baseline.zip), and [tuned XGBoost](results/xgboost_tuned.zip) include fitted estimators, native models, and candidate predictions. The complete [Optuna trial history](results/optuna_trials.json) records fold groups and scores.

## Reproduce on the Omen or Aadi’s machine

Run at the repository root, using the repository dependencies plus XGBoost, LightGBM, Optuna, FastAPI, and Uvicorn. The Omen used XGBoost 3.4.1, LightGBM 4.7.0, Optuna 5.1.0, and FastAPI 0.163.2. The cache, model results, and package directories can be external to the repository.

```powershell
python -B phase3/omen/accuracy_audit/build_audit_dataset.py --output ../omen_route_first_v1
python -B phase3/omen/accuracy_audit/train_audit_boosters.py --cache ../omen_route_first_v1 --output ../omen_booster_audit_v1 --folds 3 --trials 5 --seconds 150
```

For an interrupted cache only, use `--resume` to verify and reuse any completed trip Parquet files. For a completed cache or run, choose a new output path to run again. Aadi can load `features/{train,val}/<trip_id>.parquet` and `observations/{train,val}/<trip_id>.parquet`, fit only on the training split, and pass the full validation observation ledger to the shared scoring function in `train_canonical.py`. Use the manifest's hash and exact feature order in the resulting records. The archive contract also supports model reuse through `model_registry.py`.

## Viva points

- **Why the same road can count as wrong.** Opposite directions use different arc IDs and may have different permitted transitions. Arc-set F1 and continuity measure different properties than observation accuracy.
- **Why micro and macro both matter.** Micro counts every observation equally. Trip `00000084` accounts for 40.63% of the validation observations. Macro gives its result the same weight as any other trip, and worst-trip accuracy shows failures that either average can conceal.
- **Why Bayesian tuning stays inside the 64 trips.** TPE evaluates candidate settings on disjoint trip folds. The final 16 trips measure the candidate selected by that procedure. Candidate-level binary accuracy is secondary because each observation contains multiple negatives.
- **Why use an HMM.** Classifiers supply candidate emission scores. Directed transitions and Viterbi account for sequence consistency. Route reconstruction then collapses repeated outputs and bridges short graph gaps where possible.

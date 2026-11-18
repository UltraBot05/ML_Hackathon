# PathSnap team validation audit

The replay uses the same 16 Phase 2 validation trips, 266,367 candidate rows and 39,460 observations as Review 1. The archived directed proxy covers 39,166 observations. Trip `00000084` has 16,031 observations, or 40.63% of the validation set, so the choice between observation-weighted and trip-weighted accuracy materially changes the ranking.

The original Phase 2 graph HMM leads coverage-aware micro Top-1 at **85.42%**. Aadi's Extra Trees leads the published six-model pointwise benchmark at **74.85%**. Under the same learned-emission decoder, Extra Trees reaches **82.60%**, CatBoost reaches **81.73%**, and LightGBM reaches **71.97%**. LightGBM + HMM has the best macro directed arc-set F1 in this comparison at **0.8911**, compared with **0.8877** for the original Phase 2 HMM.

## Archived proxy results

Trip 84 rates use all 16,031 observations.

| Method | Hits | Micro coverage-aware | Micro conditional | Macro coverage-aware | Macro conditional | Worst trip coverage-aware | Trip `00000084` hits and rate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Extra Trees (Aadi) | 29,537 | 74.85% | 75.41% | 81.67% | 82.24% | `00000016` 53.49% (698/1,305) | 10,022 (62.52%) |
| Logistic Regression (Aadi) | 29,352 | 74.38% | 74.94% | 80.57% | 81.12% | `00000061` 54.27% (776/1,430) | 10,216 (63.73%) |
| CatBoost (Aadi) | 29,060 | 73.64% | 74.20% | 81.32% | 81.88% | `00000061` 51.82% (741/1,430) | 9,610 (59.95%) |
| Random Forest (Aadi) | 28,778 | 72.93% | 73.48% | 82.35% | 82.93% | `00000061` 52.17% (746/1,430) | 8,990 (56.08%) |
| XGBoost (Omen) | 27,459 | 69.59% | 70.11% | 81.87% | 82.44% | `00000084` 48.05% (7,703/16,031) | 7,703 (48.05%) |
| LightGBM (Adarsh) | 27,283 | 69.14% | 69.66% | 81.71% | 82.28% | `00000084` 47.00% (7,535/16,031) | 7,535 (47.00%) |
| Extra Trees (Adarsh) | 29,564 | 74.92% | 75.48% | 81.71% | 82.28% | `00000016` 53.49% (698/1,305) | 10,038 (62.62%) |
| Extra Trees + HMM (Aadi) | 32,593 | 82.60% | 83.22% | 87.08% | 87.69% | `00000061` 55.80% (798/1,430) | 11,945 (74.51%) |
| Extra Trees + HMM (Adarsh) | 32,528 | 82.43% | 83.05% | 87.08% | 87.69% | `00000061` 55.73% (797/1,430) | 11,875 (74.08%) |
| CatBoost + HMM | 32,249 | 81.73% | 82.34% | 87.03% | 87.64% | `00000061` 54.55% (780/1,430) | 11,569 (72.17%) |
| LightGBM + HMM | 28,400 | 71.97% | 72.51% | 85.67% | 86.26% | `00000084` 48.08% (7,708/16,031) | 7,708 (48.08%) |
| Original Phase 2 graph HMM | 33,707 | 85.42% | 86.06% | 83.67% | 84.27% | `00000016` 48.66% (635/1,305) | 13,933 (86.91%) |

The original graph HMM gets 6,225 more hits than LightGBM + HMM on trip `00000084`. LightGBM + HMM gets 918 more hits across the other 15 trips combined. The result reverses under equal trip weights.

| Method | Raw continuity | Bridged continuity | Route F1 | Route Jaccard |
| --- | ---: | ---: | ---: | ---: |
| Extra Trees (Aadi) | 71.06% | 88.94% | 0.7893 | 0.6667 |
| CatBoost (Aadi) | 68.83% | 87.47% | 0.7819 | 0.6566 |
| LightGBM (Adarsh) | 73.11% | 91.39% | 0.8042 | 0.6880 |
| Extra Trees + HMM (Aadi) | 86.11% | 99.31% | 0.8828 | 0.8004 |
| CatBoost + HMM | 84.65% | 99.21% | 0.8808 | 0.7966 |
| LightGBM + HMM | 85.23% | 99.15% | 0.8911 | 0.8146 |
| Original Phase 2 graph HMM | — | 99.17% | 0.8877 | 0.8077 |

The saved Phase 2 CSV reports bridged continuity. The replay reproduces all 32 LightGBM and LightGBM + HMM trip rows across rounded conditional Top-1, raw and bridged continuity, route F1 and Jaccard. Regenerated candidate keys, ordering, 22 float32 features and proxy IDs match the archives on every trip.

## Route-first proxy rescore

The same legacy-trained predictions score **83.54%** for CatBoost + HMM, **82.60%** for Aadi Extra Trees + HMM and **73.08%** for LightGBM + HMM against the route-first proxy. The coverage-aware denominator remains 39,460 and the conditional denominator is 39,079. Their route metrics are identical because both scores use the same predictions and reference routes.

| Method | Hits | Micro coverage-aware | Macro coverage-aware | Trip `00000084` hits and rate |
| --- | ---: | ---: | ---: | ---: |
| Extra Trees + HMM (Aadi) | 32,592 | 82.60% | 87.02% | 11,985 (74.76%) |
| CatBoost + HMM | 32,966 | 83.54% | 87.21% | 12,339 (76.97%) |
| LightGBM + HMM | 28,839 | 73.08% | 85.73% | 8,206 (51.19%) |

## Scoring contract

Coverage-aware micro Top-1 divides the sum of exact directed proxy hits by all 39,460 observations. Conditional micro Top-1 divides by the 39,166 observations with a positive candidate. Macro Top-1 averages the corresponding 16 trip percentages with equal trip weights. The worst trip column uses coverage-aware Top-1. Phase 2 derives the observation proxy from a supplied, untimestamped route sequence and the candidate pool, so a hit means agreement with that proxy.

Route metrics use the same Phase 2 postprocessing for every replayed prediction. It collapses adjacent duplicate arcs, then bridges up to five arcs and 600 m via directed Dijkstra paths. Bridged continuity measures the share of connected adjacent arc pairs. Route F1 and Jaccard compare sets of directed arcs with the supplied route. These route metrics give each trip equal weight. Arc-set overlap does not measure traversal order.

Adarsh's Viterbi decoder applies a shared directed graph transition penalty to each model's probabilities. Every model starts with the same Phase 2 graph cache state for that trip. Phase 2 feature generation warms the cache at 2 km, and its cache is keyed by source node. Preserving this state reproduces the archived Viterbi run, where the decoder itself requests a 1.5 km cutoff.

## Published score corrections

- Adarsh's training evaluator computes LightGBM's 69.14% pointwise Top-1 over all observations. The six-model leaderboard treats that rate as conditional and enters 27,079 hits. Exact first-maximum replay gives 27,283 hits, or 69.14% coverage-aware and 69.66% conditional.
- The standalone CatBoost JSON assigns its 74.20% conditional rate to the coverage-aware field. Its 29,060 hits yield 73.64% coverage-aware and 74.20% conditional, as shown in the six-model CSV.
- The original Phase 2 graph HMM's 83.22% published Top-1 averages all 80 trips. Its 16-trip validation subset averages 84.27% conditional. Each archived validation score yields one integer hit count at the stored precision. Adarsh's LightGBM+HMM CSV left two possible hit counts for trip `00000084`. Replaying that trip resolves the count to 7,708, or 48.08% coverage-aware.

## Reproduce

Use the saved Phase 2 candidate features, the original validation trip files and the team commits with their saved models. From the canonical repository, `audit_team.py` reads the artifacts and writes one JSON result. `--route-first` optionally scores the same frozen outputs against the separate route-first validation cache. The JSON records exact counts, route metrics and both proxy definitions in distinct fields.

```powershell
$env:PYTHONPATH = '../Hackathon/.pathsnap_deps;../Hackathon/.pathsnap_boost_deps'
python -B -X utf8 phase3/omen/accuracy_audit/audit_team.py --output '../Hackathon/audit_team_replay.json' --route-first '../Hackathon/omen_route_first_v1'
```

Artifacts are pinned to Aadi commit `9ab819e0c88d079cc7361a2c4c1d047746d946bc`, Adarsh commit `768a80b6100ca4332ce7d27b5736b64ed64cbac4`, and canonical Phase 2 files available at Omen commit `37099cee2786d8168a53444a785cdea299e671c2`. The CatBoost replay evaluates only the supplied numerical, symmetric-tree CBM1 model using NumPy because the native CatBoost package is absent. The model fields follow CatBoost's serialization schema. [Model schema](https://github.com/catboost/catboost/blob/master/catboost/libs/model/flatbuffers/model.fbs), [feature schema](https://github.com/catboost/catboost/blob/master/catboost/libs/model/flatbuffers/features.fbs).

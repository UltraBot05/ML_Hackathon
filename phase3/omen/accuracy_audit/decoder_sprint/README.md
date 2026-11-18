# PathSnap decoder sprint

The working API and all trained classifiers are preserved at `4b9a783734019ec21dbaf1868930dacb57f35767`. This directory is a separate decoder experiment. Directed micro Top-1 counts exact arc-ID hits over every GPS observation. Reference proxies come from the fixed candidate-independent alignment in `omen-phase2-route-first-v1`.

## Same-contract baselines and ablation

All methods used the same 16 validation trips and candidate pool. The ceiling is 39,079/39,460 (99.03%).

| Method | Directed hits | Micro Top-1 | Trip-macro Top-1 | Worst trip |
| --- | ---: | ---: | ---: | --- |
| Exact Phase 2 geometry HMM | 34,091 | 86.39% | 84.67% | 00000016, 53.49% |
| CatBoost + existing HMM | 32,966 | 83.54% | 87.21% | 00000061, 56.43% |
| Aadi Extra Trees + existing HMM | 32,592 | 82.60% | 87.02% | 00000061, 57.34% |
| Omen XGBoost + existing HMM | 29,922 | 75.83% | 86.53% | 00000084, 57.24% |
| Adarsh LightGBM + existing HMM | 28,839 | 73.08% | 85.73% | 00000084, 51.19% |
| **Frozen geometry + smoothed heading** | **34,918** | **88.49%** | **87.60%** | **00000016, 59.69%** |

The vectorized original and cutoff-only conditions both produced 34,090 hits. The exact historical replay yielded one additional match on trip 84. The cutoff regression matters for correctness even though its isolated score effect was zero in this experiment. Strict partial arcs with a 1 m projection tolerance dropped to 30,733 hits (77.88%). Adding robust emissions and stationary/gap handling in that strict chain scored 30,349 hits (76.91%). With an 8 m same-arc jitter allowance and heading evidence, the bounded projection variant recovered to 34,343 hits (87.03%). The best calibrated CatBoost blend reached 34,221 (86.72%). Each preset used the same candidate and label lists.

CatBoost temperature 0.75 was fitted by group-normalized candidate log loss on training trip `00000012`. The freeze used micro Top-1 on training trips `00000002`, `00000003`, and `00000033`. The chosen decoder uses the corrected cutoff cache, a distance emission with sigma 6 m, smoothed GPS heading weighted by motion reliability, and beta 40 m for the legacy arc-distance transition. It got 7,624/13,908 development hits. Candidate retrieval there was 66.37%, heavily influenced by training trip 33. Existing trained boosters had already seen those training trips, so the selection exercise is an ablation and not a generalization estimate. The same configuration was frozen before both raw dataset evaluations.

The new routing cache filters every query against its actual cutoff. Regression tests exercise both orders. The experimental routing graph preserves parallel arcs with their original candidate arc IDs. There are 932 redundant directed endpoint pairs among the validation graphs. Arcs with identical directed endpoints have identical geometric weights, so the preserved parallel edges leave shortest-distance semantics unchanged. The projection-aware implementation handles forward motion, bounded observation jitter, directed cycles, and disconnected endpoints. The frozen decoder retains the legacy movement formula because projection-aware variants gave lower directed micro scores.

## Why trip 00000084 was difficult

Trip 84 contributes 16,031 GPS observations (40.63% of validation), with 78 reference proxies missing from the shared pool. Exact geometry recorded 14,143 hits and 792 exact reverse-arc errors. CatBoost, Extra Trees and XGBoost had 2,682, 3,028 and 5,852 exact reverse errors respectively. Wrong roads, retrieval misses, and opposite directions are separated in [trip diagnostics](results/sprint_baseline_results.json). Within 1,089 low-net-motion windows below 0.8 m/s, median single-step speed was 6.94 m/s. That shows substantial local cancellation consistent with GPS jitter. The GPS data alone do not establish physical stops or U-turns.

The frozen heading method reached **14,352/16,031 (89.53%)** on trip 84. It made 337 fewer reverse errors than vectorized original geometry across all validation trips. The union of the five original methods' correct decisions is 36,807/39,460 (93.28%). That is a diagnostic that uses labels to count complementary hits and is unsuitable as an inference rule.

## Frozen raw-trip results

| Dataset | Directed hits | Micro Top-1 | Trip-macro Top-1 | Route F1 | Mean continuity | Replay wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 34,918 / 39,460 | 88.49% | 87.60% | 0.9337 | 99.17% | 35.55 s |
| Original student test | 16,946 / 19,160 | 88.44% | 89.15% | 0.9383 | 99.53% | 22.46 s |

Parallel-arc verification preserved the exact directed counts. The same frozen configuration ran on the lecturer-authorized local root, with its results stored outside Git. The production API remains at the known-good configuration.

## Reproduce from the checkout root

```powershell
$sprint = 'phase3/omen/accuracy_audit/decoder_sprint'
$cache = '../Hackathon/omen_route_first_v1'
$deps = (Resolve-Path '../Hackathon').Path
$env:PYTHONPATH = "$deps/.pathsnap_deps;$deps/.pathsnap_boost_deps"
python -B "$sprint/run_decoder_sprint.py" --repo . --cache $cache --stage val --frozen "$sprint/results/sprint_frozen_v2.json" --only-frozen --output '../validation_decoder_replay.json'
python -B "$sprint/run_decoder_sprint.py" --repo . --cache $cache --stage heldout --frozen "$sprint/results/sprint_frozen_v2.json" --root Dataset/student_test --output '../student_decoder_replay.json'
python -B "$sprint/sprint_baseline.py" --cache $cache --xgb-archive phase3/omen/accuracy_audit/results/xgboost_baseline.zip --output '../legacy_decoder_replay.json'
$env:PATHSNAP_REPO = (Get-Location).Path
python -B -m unittest discover -s $sprint -p test_sprint_decoder.py -v
```

For an authorized local evaluation root, run the same heldout command with its root and a separate external output path. Add `--private` to suppress trip identifiers in console output. Raw prediction reads only track, node, and arc files. The reference route is loaded in the scorer after the predictions exist. The detailed lecturer results stay out of Git.

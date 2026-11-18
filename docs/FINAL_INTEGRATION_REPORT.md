# PathSnap team report

## Contributions and provenance

- **Adarsh** developed the Phase 1 exploration and Phase 2 candidate engineering, the canonical feature split, Extra Trees and LightGBM experiments, and directed HMM integration.
- **Aadi** developed the Random Forest and CatBoost experiments, the six-model score audit, and the Streamlit and Folium evaluator.
- **Omen** developed the preprocessing and label audit, Logistic Regression and XGBoost baselines, the controlled booster comparison and Optuna search, decoder experiments, and backend integration.

The final integration branch combines the team's work. The final demo bridge runs a separately authorized TA-provided inference engine locally. Its 24-feature tuned LightGBM and decoding path are distinct from the team's trained 22-feature classifiers. The bridge and UI were implemented and tested by the team. The TA runtime, weights, lecturer dataset, and detailed reference outputs remain outside Git.

## Comparable team experiments

The original Phase 2 validation protocol uses 64 training trips and 16 validation trips, with 39,460 validation observations. Micro directed Top-1 counts exact arc IDs across every observation. The Phase 2 candidate-window proxy and the later independent route-first proxy are distinct reference contracts.

| Model | Original Phase 2 proxy micro Top-1 |
| --- | ---: |
| Aadi Extra Trees | 74.85% |
| Aadi Logistic Regression | 74.38% |
| CatBoost | 73.64% |
| Random Forest | 72.93% |
| Omen XGBoost | 69.59% |
| Adarsh LightGBM | 69.14% |

The earlier LightGBM leaderboard row applied coverage twice. The recovered hit count is 27,283 of 39,460. The [team audit](../phase3/omen/accuracy_audit/AUDIT_TEAM.md) records the exact replay, the corresponding conditional and macro results, and the reconstruction metrics.

Under the independent route-first proxy, the Omen XGBoost baseline reached 71.01% candidate-selection micro Top-1 and LightGBM reached 70.47%. Five training-only Optuna trials selected an XGBoost configuration that reached 70.93% on validation. The [route-first audit](../phase3/omen/accuracy_audit/README.md) records the model artifacts and trial history. The geometry-and-heading decoder sprint reached 88.49% on canonical validation and 88.44% on the original student-test trips under its independent proxy. The [decoder report](../phase3/omen/accuracy_audit/decoder_sprint/README.md) gives the counts and failure analysis.

## Integration and limitations

The demo pipeline routes Aadi's trip selection to the Omen adapter and returns the engine's directed route segments and measured timing. A live 20-trip regression compares the adapter with fresh reference requests and verifies the ordered route IDs and geometry. Detailed reference output belongs in a private local verification record. The [server guide](../server/README.md) describes the run configuration and request contract.

The original team model reports remain the evidence for team training and ablation. The local reference runtime is a separate inference dependency and has a separate scoring implementation. A self-contained team model at that accuracy still needs independent feature and decoder implementation, reproducible training, and verification.

### Self-contained replacement

1. Implement dynamic local UTM projection, turnaround-aware GPS heading, and the eight-candidate directed arc search in team-owned modules.
2. Version the 24 feature columns and order. Train a separate team-owned LightGBM from the permitted training partition, recording the split, seed, parameters, trials, and model hash.
3. Integrate a hybrid HMM with directed reconstruction and observation realignment along its predicted route. Keep reference-route access in the scoring stage.
4. Replay both evaluation roots with the new artifacts and compare exact directed counts before switching the adapter's default.

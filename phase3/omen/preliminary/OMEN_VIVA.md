# PathSnap Omen viva notes

## Data and labels

The PES dataset handout defines `.track` as longitude, latitude, and seconds. The row index in `.nodes` is the node ID. The row index in `.arcs` is the directed arc ID and its two columns are endpoint node IDs. A `.route` is an ordered sequence of arc IDs. GPS rows and route entries use different units of indexing.

One training row describes a candidate road arc for a single GPS observation. The target is 1 when the arc is the proxy reference for that observation and 0 otherwise. We derive the proxy by minimizing the sum of GPS-to-segment distances while enforcing nondecreasing positions in the ordered reference route. Repeated arc IDs are allowed at different route positions. We flag geometric ambiguity when at least two distinct reference arcs lie within 2 m of the shortest GPS-to-reference-arc distance. Direction and localization ambiguity can still affect these labels.

The candidate generator and all features use GPS and the road network. The feature extractor finishes before opening the reference route. The classifier sees only the named columns in its feature contract. IDs and the target label are excluded. We project around each trip's center to approximate distances in metres. The KD-tree searches arc midpoints in segment-length buckets, and exact point-to-segment distances filter the resulting hits. The initial settings retain the eight closest arcs inside 75 m.

The baseline columns are distance to the segment, heading difference, segment length, GPS speed, timestamp gap, and distance rank. The contextual experiment adds candidate position along the segment, distances to its endpoints, two GPS-window heading differences, window speed, distances from earlier and later GPS points to the segment, and three directed-network costs. Those costs measure how well a candidate fits the GPS sequence and its incoming and outgoing candidates. They use the observed GPS step, same-arc direction, and whether endpoints connect. Computation restarts across long GPS gaps.

## Models

Logistic Regression fits a linear score over standardized features and maps the score through the sigmoid, `p = 1 / (1 + exp(-z))`. Its binary cross-entropy penalizes confident wrong predictions and its L2 regularization discourages large coefficients. `C` is the inverse regularization strength. A single pipeline fits the scaler only within the training partition of each fold. The baseline uses LBFGS, `C=1`, and a 1,000-iteration limit. We kept class weights at their defaults for the probability baseline.

XGBoost adds trees sequentially. Each new tree uses gradients and curvature of the binary logistic loss to improve the current predictions. Maximum depth limits interaction complexity, the learning rate shrinks each tree's contribution, and the estimator count controls the number of boosting rounds. Row and column subsampling limit how much data each tree sees. XGBoost can also regularize leaf weights and require a minimum improvement to split. The baseline uses CPU histogram training, 150 trees, depth 5, learning rate 0.08, and 0.85 for both sampling fractions.

Random Forest averages independently fitted randomized trees. LightGBM is another gradient boosting implementation and commonly grows trees leaf-wise. CatBoost is gradient boosting with machinery for categorical features and ordered boosting. PathSnap's current feature columns are numerical. Choose among the team models using measured performance on identical features, groups, and metrics.

## Validation and measured baseline

The first experiment fitted trips `00000003`–`00000006` and held out `00000001` and `00000002`. The holdout had 8,558 observations and 60,925 candidate rows. Candidate miss rate was 10.04%, limiting coverage-aware Top-1 to 89.96% even if ranking were perfect. The two models reached 52.03% and 51.90% coverage-aware Top-1, respectively.

The expanded baseline used three-fold GroupKFold on the 80 training trips; each trip belongs to one validation fold. Both models were then refitted on all 80 trips, with 208,417 GPS observations and 1,242,266 candidate rows. The ten student-test trips held 19,160 observations and 125,556 candidates. The student-test candidate miss rate was 0.39%. The evaluator partition is reserved for final evaluation.

| Model | CV macro F1 | Student candidate accuracy | Student coverage-aware Top-1 | Student macro F1 | Student log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8439 | 92.80% | 79.98% | 0.8559 | 0.1933 |
| XGBoost | 0.8573 | 93.33% | 80.20% | 0.8673 | 0.1625 |

Candidate accuracy and macro F1 classify individual candidate rows at a 0.5 probability threshold. The many negative candidates make row accuracy a weak measure of matching quality. Top-1 selects the highest-scored candidate once per GPS observation. Coverage-aware Top-1 counts misses in its denominator. Conditional Top-1 divides by the number of observations where the reference proxy is a candidate. The results quantify agreement with proxy labels.

The 17-feature contextual experiment used the same candidate rows, labels, and folds.

| Model | CV macro F1 | Student candidate accuracy | Student coverage-aware Top-1 | Student macro F1 | Student log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9371 | 96.53% | 90.30% | 0.9330 | 0.0971 |
| XGBoost | 0.9571 | 97.70% | 92.29% | 0.9549 | 0.0606 |

For contextual XGBoost, 752 of 1,478 directed student-test errors chose the same endpoints in reverse order. Trip `00000077` contributed 561 reverse errors. Among those, smoothed GPS motion aligned with the prediction for 372 observations and with the proxy for 189. Counting exact reverse geometry as correct gives a separate 96.21% diagnostic. Exact directed matching remains the Top-1 metric in the team summary.

The motion-aware training label experiment added a GPS direction penalty to the ordered alignment during fitting. All validation folds and student-test metrics kept the original geometric proxy labels. Student-test Top-1 changed to 90.61% for Logistic Regression and 91.87% for XGBoost.

The 20-feature directed-bridge experiment added three cost columns from transitions that can traverse up to two intervening arcs. It used the original geometric labels and the same 75 m, eight-candidate pool. Logistic Regression reached 90.48% student-test Top-1 and 0.9376 CV macro F1. XGBoost reached 92.61% student-test Top-1 and 0.9573 CV macro F1. Of its 1,416 directed errors, 754 matched the same endpoints in reverse. That yields a separate 96.54% diagnostic with direction ignored. The artifacts and team-format JSON summaries are in `pathsnap_bridge_experiment`.

Candidate generation also limits training-set coverage. In trips `00000033` and `00000054`, the median distance to the nearest reference-route segment was 122 m and 139 m. Widening from 75 m and eight candidates to 300 m and 32 candidates raised coverage from 33.40% to 47.30% and from 9.76% to 76.78% on those trips. The contextual model comparison kept the original settings to measure the effect of the new features.

Experiment archives contain `results.json`, a fitted `model.joblib`, validation probabilities per candidate, and validation proxy labels and coverage per observation. The team JSON summary records mean grouped-CV macro F1, coverage-aware student-test Top-1, and model-only times. It uses JSON null for metrics the run did not measure. The six-trip holdout has null CV and student-test metrics. The exact split IDs and confusion matrices are in `results.json`.

## Bayesian optimisation and route reconstruction

Select a model once the team's comparable results are available. Use Optuna's TPE sampler with fixed trip folds, record every trial, then refit the selected configuration on the 80 trips. Evaluate the tuned and untuned fits on the same student-test trips. The preliminary models are untuned.

For HMM integration, a hidden state is a road candidate for one observation. Its emission score comes from the classifier. A transition reflects network connectivity and plausibility of the travel distance or time between GPS observations. Viterbi keeps the best cumulative score ending at each candidate and backtracks to obtain a route. A production decoder must handle empty candidate sets, impossible transitions, and GPS gaps. Compare the classifier alone against the decoded route on pointwise agreement and route continuity.

## Running locally

From the workspace, with the local package targets available in this environment:

```powershell
$env:PYTHONPATH = (Resolve-Path .pathsnap_deps).Path + ';' + (Resolve-Path .pathsnap_api_deps).Path
python -B pathsnap_quickstart.py --output another_proxy_run --trips 80 --cv-folds 3 --student-test 'Student folder/Student folder/student_test'
python -B -m unittest -v test_pathsnap
$env:PATHSNAP_SPLIT = 'student_test'
$env:PATHSNAP_MODELS_JSON = '{"logistic":"pathsnap_bridge_experiment/logistic.zip","xgboost":"pathsnap_bridge_experiment/xgboost.zip"}'
python -B pathsnap_api.py
```

Use `--context` to include the contextual feature columns, `--bridges` to add up to two intervening directed arcs to the transition evidence, and `--directional-labels` to try the motion-aware training labels. Choose a fresh output directory for each experiment. Set `PATHSNAP_DATA_ROOT` to the directory containing the chosen trip partition. `PATHSNAP_HOST` and `PATHSNAP_PORT` default to `0.0.0.0` and `8000`. The Omen's verified ZeroTier address is `10.92.151.151`. If the evaluator UI runs in a browser on another origin, set `PATHSNAP_CORS_ORIGINS` to that origin. The API lists ID-only trips and reports health. `POST /predict` returns HTTP 503 until the canonical feature pipeline and route decoder are connected. Keep evaluator reference routes on the backend and outside the prediction path.

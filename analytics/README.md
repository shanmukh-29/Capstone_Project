# Analytics Pipeline

Run `py analytics/analysis.py`. The script loads Titanic once from `sns.load_dataset('titanic')` when `analytics/titanic.csv` does not exist, immediately saves that offline fallback, then performs profiling, threshold-based cleaning, IQR outlier counts, four EDA charts plus a six-column correlation heatmap, exploratory standardization, stratified modeling, model metrics, a Random Forest grid search with OOB scoring, and linear regression.

Artifacts are written to `analytics/figures/`, `eda_report.txt`, `classification_metrics.csv`, and `best_pipeline.joblib`. The saved joblib artifact contains the fitted preprocessing transformer and estimator together and is reloaded against raw test rows in the script.

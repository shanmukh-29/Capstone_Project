"""Single-load Titanic EDA and modeling pipeline."""
from pathlib import Path
import json
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
RANDOM_STATE = 42


def load_once():
    csv_path = ROOT / "titanic.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    data = sns.load_dataset("titanic")
    data.to_csv(csv_path, index=False)
    return data


def iqr_count(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


def main():
    df = load_once()
    with (ROOT / "eda_report.txt").open("w", encoding="utf-8") as report:
        report.write(str(df.info(buf=None)) + "\n")
        report.write(f"shape={df.shape}\n{df.describe(include='all').to_string()}\n")
        missing = (df.isna().mean() * 100).loc[lambda values: values > 0]
        report.write(f"missing_percentages={missing.to_dict()}\n")
        report.write("Rule: under 5% missing rows are dropped; 5%-30% are median/mode imputed; high-missing deck is encoded as Unknown.\n")
        for column in missing.index:
            rate = missing[column]
            if rate < 5:
                report.write(f"{column}: {rate:.2f}% -> drop affected rows\n")
            elif rate <= 30:
                report.write(f"{column}: {rate:.2f}% -> impute\n")
            else:
                report.write(f"{column}: {rate:.2f}% -> retain as Unknown category or median-impute numeric\n")
        df = df.dropna(subset=["survived", "pclass", "sex", "fare"]).copy()
        df["age"] = df["age"].fillna(df["age"].median())
        df["embarked"] = df["embarked"].fillna("Unknown")
        df["deck"] = df["deck"].fillna("Unknown")
        for column in ("age", "fare"):
            before = df[column].describe()[["mean", "std"]]
            df[column + "_z"] = (df[column] - df[column].mean()) / df[column].std()
            after = df[column + "_z"].describe()[["mean", "std"]]
            report.write(f"{column} outliers by IQR={iqr_count(df[column])}; before={before.to_dict()}; after={after.to_dict()}\n")
        fare_mode = df["fare"].mode().iloc[0]
        report.write(f"fare mean={df.fare.mean():.3f}, median={df.fare.median():.3f}, mode={fare_mode:.3f}; the mean > median > mode ordering indicates right skew.\n")
        report.write("survival_by_sex\n" + df.groupby("sex")["survived"].mean().to_string() + "\n")
        report.write("survival_by_pclass\n" + df.groupby("pclass")["survived"].mean().to_string() + "\n")
        report.write("survival_by_sex_pclass\n" + df.groupby(["sex", "pclass"])["survived"].mean().to_string() + "\n")
        corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
        corr = df[corr_cols].corr()
        report.write("exact_6_column_correlation\n" + corr.to_string() + "\n")
        pairs = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool)).stack().abs().sort_values(ascending=False).head(2)
        report.write(f"two_strongest_absolute_pairs={pairs.to_dict()}\n")
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        sns.histplot(df["age"], ax=axes[0, 0], kde=True); axes[0, 0].set_title("Age distribution")
        sns.boxplot(x=df["age"], ax=axes[0, 1]); axes[0, 1].set_title("Age outliers")
        sns.histplot(df["fare"], ax=axes[1, 0], kde=True); axes[1, 0].set_title("Fare distribution")
        sns.boxplot(x=df["fare"], ax=axes[1, 1]); axes[1, 1].set_title("Fare outliers")
        fig.tight_layout(); fig.savefig(FIGURES / "univariate.png"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 5)); sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax); fig.tight_layout(); fig.savefig(FIGURES / "correlation_heatmap.png"); plt.close(fig)
        charts = [
            (sns.barplot, {"data": df, "x": "sex", "y": "survived", "hue": "pclass"}, "survival_by_sex_class.png"),
            (sns.boxplot, {"data": df, "x": "survived", "y": "fare"}, "fare_by_survival.png"),
            (sns.scatterplot, {"data": df, "x": "age", "y": "fare", "hue": "survived"}, "age_fare_survival.png"),
            (sns.barplot, {"data": df, "x": "pclass", "y": "survived", "hue": "sex"}, "class_sex_survival.png"),
        ]
        for chart, kwargs, filename in charts:
            fig, ax = plt.subplots(figsize=(8, 5)); chart(ax=ax, **kwargs); fig.tight_layout(); fig.savefig(FIGURES / filename); plt.close(fig)
        X = df[["pclass", "age", "sibsp", "parch", "fare", "sex", "embarked"]]
        y = df["survived"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=RANDOM_STATE)
        numeric = ["pclass", "age", "sibsp", "parch", "fare"]
        categorical = ["sex", "embarked"]
        preprocessor = ColumnTransformer([("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
        models = {"Logistic Regression": LogisticRegression(max_iter=1000), "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), "Random Forest": RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE)}
        metrics = {}
        for name, estimator in models.items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            pipe.fit(X_train, y_train); predictions = pipe.predict(X_test); probabilities = pipe.predict_proba(X_test)[:, 1]
            metrics[name] = {"accuracy": accuracy_score(y_test, predictions), "precision": precision_score(y_test, predictions), "recall": recall_score(y_test, predictions), "f1": f1_score(y_test, predictions), "auc": roc_auc_score(y_test, probabilities)}
            report.write(f"{name} confusion_matrix={confusion_matrix(y_test, predictions).tolist()}\n")
            if name == "Decision Tree":
                feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
                fig, ax = plt.subplots(figsize=(18, 9)); plot_tree(pipe.named_steps["model"], feature_names=feature_names, class_names=["0", "1"], filled=True, ax=ax); fig.savefig(FIGURES / "decision_tree.png"); plt.close(fig)
        comparison = pd.DataFrame(metrics).T
        comparison.to_csv(ROOT / "classification_metrics.csv")
        comparison.to_string(buf=report)
        rf_search = GridSearchCV(Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(oob_score=True, random_state=RANDOM_STATE, bootstrap=True))]), {"model__n_estimators": [100, 150], "model__max_depth": [None, 5], "model__max_features": ["sqrt", "log2"]}, cv=3, scoring="f1")
        rf_search.fit(X_train, y_train)
        report.write(f"grid_best={rf_search.best_params_}; oob_score={rf_search.best_estimator_.named_steps['model'].oob_score_}\n")
        best_pipeline = Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE))]).fit(X_train, y_train)
        joblib.dump(best_pipeline, ROOT / "best_pipeline.joblib")
        raw_prediction = joblib.load(ROOT / "best_pipeline.joblib").predict(X_test.head(1))
        report.write(f"reloaded_raw_prediction={raw_prediction.tolist()}\n")
        regression_features = df[["survived", "pclass", "age", "sibsp", "parch"]]
        reg = LinearRegression().fit(regression_features, df["fare"])
        residuals = df["fare"] - reg.predict(regression_features); rmse = mean_squared_error(df["fare"], reg.predict(regression_features)) ** .5; r2 = r2_score(df["fare"], reg.predict(regression_features)); n, p = len(df), regression_features.shape[1]
        adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        report.write(f"regression MAE={mean_absolute_error(df.fare, reg.predict(regression_features)):.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, adjusted_R2={adjusted:.3f}; residual plot should be inspected for non-random spread/heteroscedasticity.\n")
        fig, ax = plt.subplots(); sns.scatterplot(x=reg.predict(regression_features), y=residuals, ax=ax); ax.axhline(0, color="red"); ax.set(xlabel="Predicted fare", ylabel="Residual"); fig.savefig(FIGURES / "regression_residuals.png"); plt.close(fig)
    print("Analytics complete. See analytics/eda_report.txt and analytics/figures/")


if __name__ == "__main__":
    main()

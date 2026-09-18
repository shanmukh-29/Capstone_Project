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


def main():
    # Load and preprocess data
    df = pd.read_csv(ROOT / "titanic.csv")
    df["deck"] = df["deck"].astype("category")
    # Add 'Unknown' to the categories before filling NaN values
    if "Unknown" not in df["deck"].cat.categories:
        df["deck"] = df["deck"].cat.add_categories("Unknown")
    df["deck"] = df["deck"].fillna("Unknown")
    df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
    df["age"] = df["age"].fillna(df["age"].median())

    # Define features and target
    numerical_features = ["age", "fare", "sibsp", "parch"]
    categorical_features = ["pclass", "sex", "embark_town", "deck"]

    # Create preprocessor
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE)
    }

    # Split data
    X = df.drop("survived", axis=1)
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # Train and evaluate models
    with (ROOT / "eda_report.txt").open("w", encoding="utf-8") as report:
        report.write("# Titanic EDA and Modeling Report\n\n")
        report.write(f"## Data Info\n\n{df.info(buf=None)}\n\n")
        report.write(f"## Describe Numerical Features\n\n{df[numerical_features].describe().to_string()}\n\n")
        report.write(f"## Describe Categorical Features\n\n")
        for col in categorical_features:
            report.write(f"### {col}\n\n{df[col].value_counts().to_string()}\n\n")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df["age"], kde=True, ax=axes[0])
        axes[0].set_title("Age Distribution")
        sns.countplot(x="survived", data=df, ax=axes[1])
        axes[1].set_title("Survival Count")
        fig.tight_layout()
        fig.savefig(FIGURES / "distributions.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(x="survived", y="age", data=df, ax=axes[0])
        axes[0].set_title("Survival by Age")
        sns.boxplot(x="survived", y="fare", data=df, ax=axes[1])
        axes[1].set_title("Survival by Fare")
        fig.tight_layout()
        fig.savefig(FIGURES / "survival_by_features.png")
        plt.close(fig)

        metrics = {}
        for name, model in models.items():
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            predictions = pipe.predict(X_test)
            probabilities = pipe.predict_proba(X_test)[:, 1]
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

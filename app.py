#prediction model
#determining the data type
#determinig the ML task
#selecting the model
#calculating the accuracy
#apllying a threshold (based on model of prediction)
#accepting or rejecting
#prediction result (estimation)
#graph these predictions(plotting the results)
#explaining the result
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_categorical_dtype

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from xgboost import XGBClassifier, XGBRegressor

import streamlit as st
import matplotlib.pyplot as plt


@dataclass
class ClassificationMetrics:
    accuracy: float
    f1_macro: float
    confusion_matrix: np.ndarray


@dataclass
class RegressionMetrics:
    rmse: float
    mae: float
    r2: float


MetricsType = Union[ClassificationMetrics, RegressionMetrics]


@dataclass
class ModelResult:
    name: str
    model: Any
    task_type: str
    metrics: MetricsType


@dataclass
class SelectorSummary:
    task_type: str
    best_model_name: str
    best_model_metrics: Dict[str, Any]
    threshold: float
    accepted: bool


class PredictiveModelSelector:
    def __init__(
        self,
        threshold_classification: float = 0.70,
        threshold_regression: float = 0.70,
        test_size: float = 0.25,
        random_state: int = 42,
    ) -> None:
        self.threshold_classification = threshold_classification
        self.threshold_regression = threshold_regression
        self.test_size = test_size
        self.random_state = random_state

        self.task_type: Optional[str] = None
        self.target_name: Optional[str] = None
        self.preprocessor: Optional[ColumnTransformer] = None

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self.model_results: List[ModelResult] = []
        self.best_result: Optional[ModelResult] = None

        self.class_labels_: Optional[np.ndarray] = None

    @staticmethod
    def detect_task_type(y: pd.Series) -> str:
        unique_vals = int(y.nunique())

        if is_categorical_dtype(y) or y.dtype == "object":
            return "classification"

        if is_integer_dtype(y) and unique_vals <= 20:
            return "classification"

        return "regression"

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        numeric_transformer = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="passthrough",
        )
        return preprocessor

    def _get_classification_models(self) -> Dict[str, Pipeline]:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor is not initialized.")

        base_models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
            ),
            "XGBClassifier": XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=self.random_state,
            ),
        }

        models: Dict[str, Pipeline] = {}
        for name, est in base_models.items():
            models[name] = Pipeline(
                steps=[
                    ("preprocess", self.preprocessor),
                    ("model", est),
                ]
            )
        return models

    def _get_regression_models(self) -> Dict[str, Pipeline]:
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor is not initialized.")

        base_models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=200,
                random_state=self.random_state,
            ),
            "XGBRegressor": XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                random_state=self.random_state,
            ),
        }

        models: Dict[str, Pipeline] = {}
        for name, est in base_models.items():
            models[name] = Pipeline(
                steps=[
                    ("preprocess", self.preprocessor),
                    ("model", est),
                ]
            )
        return models

    def _get_split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if (
            self.X_train is None
            or self.X_test is None
            or self.y_train is None
            or self.y_test is None
        ):
            raise RuntimeError("Data has not been split.")

        assert isinstance(self.X_train, pd.DataFrame)
        assert isinstance(self.X_test, pd.DataFrame)
        assert isinstance(self.y_train, pd.Series)
        assert isinstance(self.y_test, pd.Series)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def _evaluate_classification_model(
        self,
        name: str,
        model: Pipeline,
    ) -> ModelResult:
        X_train, X_test, y_train, y_test = self._get_split_data()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="macro"))
        cm = confusion_matrix(y_test, preds)

        cls_metrics = ClassificationMetrics(
            accuracy=acc,
            f1_macro=f1,
            confusion_matrix=cm,
        )

        return ModelResult(name, model, "classification", cls_metrics)

    def _evaluate_regression_model(
        self,
        name: str,
        model: Pipeline,
    ) -> ModelResult:
        X_train, X_test, y_train, y_test = self._get_split_data()

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        reg_metrics = RegressionMetrics(rmse, mae, r2)

        return ModelResult(name, model, "regression", reg_metrics)

    def train_and_evaluate(self) -> None:
        self.model_results = []

        if self.task_type == "classification":
            models = self._get_classification_models()
            for name, model in models.items():
                self.model_results.append(
                    self._evaluate_classification_model(name, model)
                )
        elif self.task_type == "regression":
            models = self._get_regression_models()
            for name, model in models.items():
                self.model_results.append(
                    self._evaluate_regression_model(name, model)
                )
        else:
            raise RuntimeError("task_type is not set.")

    def select_best_model(self) -> None:
        if not self.model_results:
            raise RuntimeError("No model_results present.")

        if self.task_type == "classification":
            def key_fn(m: ModelResult) -> float:
                cls_metrics = cast(ClassificationMetrics, m.metrics)
                return cls_metrics.f1_macro

            self.best_result = max(self.model_results, key=key_fn)

        elif self.task_type == "regression":
            def key_fn(m: ModelResult) -> float:
                reg_metrics = cast(RegressionMetrics, m.metrics)
                return reg_metrics.r2

            self.best_result = max(self.model_results, key=key_fn)
        else:
            raise RuntimeError("task_type is not set.")

    def fit(self, df: pd.DataFrame, target: str) -> None:
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found.")

        X = df.drop(columns=[target])
        y = df[target]

        self.task_type = self.detect_task_type(y)
        self.preprocessor = self.build_preprocessor(X)
        self.class_labels_ = None

        if self.task_type == "classification":
            # 1) Keep only classes with at least 2 samples
            class_counts = y.value_counts()
            sufficient = class_counts[class_counts >= 2]

            if sufficient.shape[0] < 2:
                raise ValueError(
                    "Classification requires at least 2 classes with at least 2 samples each."
                )

            valid_classes = sufficient.index
            mask = y.isin(valid_classes)

            X = X.loc[mask].copy()
            y = y.loc[mask].copy()

            # 2) Encode labels into 0..N-1 for XGBoost and others
            y_codes, uniques = pd.factorize(y)
            self.class_labels_ = uniques
            y = pd.Series(y_codes, index=y.index)

            stratify_arg = y
        else:
            stratify_arg = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_arg,
        )

        self.train_and_evaluate()
        self.select_best_model()

    def summary(self) -> SelectorSummary:
        if self.best_result is None or self.task_type is None:
            raise RuntimeError("Model not fitted yet.")

        task_type: str = self.task_type

        if task_type == "classification":
            cls_metrics = cast(ClassificationMetrics, self.best_result.metrics)
            threshold = self.threshold_classification
            score_value = cls_metrics.f1_macro

            metrics_dict: Dict[str, Any] = {
                "accuracy": cls_metrics.accuracy,
                "f1_macro": cls_metrics.f1_macro,
                "confusion_matrix": cls_metrics.confusion_matrix.tolist(),
            }
        else:
            reg_metrics = cast(RegressionMetrics, self.best_result.metrics)
            threshold = self.threshold_regression
            score_value = reg_metrics.r2

            metrics_dict = {
                "rmse": reg_metrics.rmse,
                "mae": reg_metrics.mae,
                "r2": reg_metrics.r2,
            }

        accepted = bool(score_value >= threshold)

        return SelectorSummary(
            task_type=task_type,
            best_model_name=self.best_result.name,
            best_model_metrics=metrics_dict,
            threshold=threshold,
            accepted=accepted,
        )

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        if self.best_result is None:
            raise RuntimeError("No best model available.")
        return self.best_result.model.predict(X_new)


st.set_page_config(page_title="Predictive Model Selector", layout="wide")


def main() -> None:
    st.title("📊 Predictive Model Selector (Excel / CSV, XGBoost-enabled)")

    uploaded_file = st.file_uploader(
        "Upload Excel or CSV",
        type=["xlsx", "xls", "csv"],
    )

    if uploaded_file is None:
        st.info("Upload a dataset to start.")
        return

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV loaded successfully!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return
    else:
        try:
            xls = pd.ExcelFile(uploaded_file)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return

        sheet = st.selectbox("Select sheet", xls.sheet_names)
        try:
            df = xls.parse(sheet)
            st.success(f"Excel sheet '{sheet}' loaded successfully!")
        except Exception as e:
            st.error(f"Error parsing sheet '{sheet}': {e}")
            return

    if df.empty:
        st.warning("The loaded data is empty.")
        return

    st.write("### Data preview")
    st.dataframe(df.head())

    target = st.selectbox("Select target column", df.columns)

    col1, col2 = st.columns(2)
    with col1:
        thr_cls = st.slider("Classification threshold (F1)", 0.0, 1.0, 0.70)
    with col2:
        thr_reg = st.slider("Regression threshold (R²)", 0.0, 1.0, 0.70)

    if st.button("🚀 Train Model"):
        try:
            selector = PredictiveModelSelector(
                threshold_classification=thr_cls,
                threshold_regression=thr_reg,
            )
            selector.fit(df, target)
            summary = selector.summary()
        except Exception as e:
            st.error(f"Error during training: {e}")
            return

        st.write("### Model Summary")
        st.metric("Task type", summary.task_type)
        st.metric("Best model", summary.best_model_name)
        st.metric("Accepted", "✅" if summary.accepted else "❌")
        st.code(f"Threshold used: {summary.threshold:.2f}", language="text")

        st.write("### Metrics")
        st.json(summary.best_model_metrics)

        if summary.task_type == "classification":
            cm_list = summary.best_model_metrics.get("confusion_matrix")
            if cm_list is not None:
                cm = np.array(cm_list)
                st.write("### Confusion Matrix (encoded labels 0..N-1)")
                fig, ax = plt.subplots()
                im = ax.imshow(cm, cmap="Blues")
                ax.figure.colorbar(im, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
                st.pyplot(fig)


if __name__ == "__main__":
    main()
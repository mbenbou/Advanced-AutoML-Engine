import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib
import base64
import warnings

# --- Suppress specific LightGBM/Sklearn noise warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

# --- Machine Learning Libraries ---
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, 
    mean_absolute_error, confusion_matrix
)
from xgboost import XGBClassifier, XGBRegressor

# --- Optional Imports (Robust Loading) ---
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_INSTALLED = True
except ImportError:
    IMBLEARN_INSTALLED = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LGBM_INSTALLED = True
except ImportError:
    LGBM_INSTALLED = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_INSTALLED = True
except ImportError:
    CATBOOST_INSTALLED = False


# ==========================================
# 1. CORE LOGIC: Task Detection & Data Handling
# ==========================================

class TaskDetector:
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col
        self.target_data = data[target_col]
    
    def detect(self):
        """Analyzes the target variable to determine problem type and characteristics."""
        unique_vals = self.target_data.nunique()
        total_rows = len(self.target_data)
        dtype = self.target_data.dtype
        
        info = {
            'type': 'Unknown',
            'subtype': None,
            'is_imbalanced': False,
            'imbalance_ratio': None
        }

        # --- GUARDRAIL: Check for ID-like or Date-like targets ---
        if unique_vals > 50 and (unique_vals / total_rows > 0.90):
             st.error(f"⚠️ **CRITICAL WARNING**: Target column '{self.target_col}' has {unique_vals} unique values (almost unique per row).")
             st.warning("It looks like an **ID** or **Date**. Models will fail. Please select a Category or Numerical column.")

        # --- Rule 1: Classification vs Regression ---
        if (unique_vals <= 20) or (dtype == 'object') or (dtype == 'bool'):
            info['type'] = 'Classification'
            if unique_vals == 2:
                info['subtype'] = 'Binary'
            else:
                info['subtype'] = 'Multiclass'
            
            value_counts = self.target_data.value_counts(normalize=True)
            min_class_ratio = value_counts.min()
            info['imbalance_ratio'] = min_class_ratio
            if min_class_ratio < 0.20: 
                info['is_imbalanced'] = True
        else:
            info['type'] = 'Regression'
            info['subtype'] = 'Continuous'
        
        return info

# ==========================================
# 2. CORE LOGIC: Preprocessing & Pipeline Building
# ==========================================

class PipelineBuilder:
    def __init__(self, task_info):
        self.task_info = task_info
        
    def get_preprocessor(self, X):
        """Builds the column transformer with pandas output for feature names."""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns
        
        num_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)), 
            ('scaler', RobustScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, numeric_features),
                ('cat', cat_transformer, categorical_features)
            ],
            verbose_feature_names_out=False
        )
        preprocessor.set_output(transform="pandas")
        return preprocessor

    def get_models(self):
        """Returns a dictionary of models + hyperparameter grids."""
        models = {}
        
        # --- CLASSIFICATION MODELS ---
        if self.task_info['type'] == 'Classification':
            models['Logistic Regression'] = {
                'model': LogisticRegression(max_iter=1000, solver='liblinear'),
                'params': {'model__C': [0.1, 1, 10]}
            }
            models['Random Forest'] = {
                'model': RandomForestClassifier(random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10]}
            }
            models['Extra Trees'] = {
                'model': ExtraTreesClassifier(random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10]}
            }
            models['SVM'] = {
                'model': SVC(probability=True, random_state=42),
                'params': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}
            }
            models['KNN'] = {
                'model': KNeighborsClassifier(),
                'params': {'model__n_neighbors': [3, 5, 7], 'model__weights': ['uniform', 'distance']}
            }
            models['XGBoost'] = {
                'model': XGBClassifier(eval_metric='logloss', random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
            }
            models['AdaBoost'] = {
                'model': AdaBoostClassifier(algorithm='SAMME', random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
            }
            models['Naive Bayes'] = {'model': GaussianNB(), 'params': {}}
            
            if LGBM_INSTALLED:
                models['LightGBM'] = {
                    'model': LGBMClassifier(random_state=42, verbose=-1),
                    'params': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
                }
            if CATBOOST_INSTALLED:
                models['CatBoost'] = {
                    'model': CatBoostClassifier(random_state=42, verbose=0),
                    'params': {'model__iterations': [50, 100], 'model__learning_rate': [0.01, 0.1], 'model__depth': [4, 6]}
                }

        # --- REGRESSION MODELS ---
        else:
            models['Ridge Regression'] = {
                'model': Ridge(),
                'params': {'model__alpha': [0.1, 1.0, 10.0]}
            }
            models['Random Forest'] = {
                'model': RandomForestRegressor(random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10]}
            }
            models['Extra Trees'] = {
                'model': ExtraTreesRegressor(random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10]}
            }
            models['SVR'] = {
                'model': SVR(),
                'params': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}
            }
            models['KNN'] = {
                'model': KNeighborsRegressor(),
                'params': {'model__n_neighbors': [3, 5, 7], 'model__weights': ['uniform', 'distance']}
            }
            models['XGBoost'] = {
                'model': XGBRegressor(random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
            }
            models['AdaBoost'] = {
                'model': AdaBoostRegressor(random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
            }
            if LGBM_INSTALLED:
                models['LightGBM'] = {
                    'model': LGBMRegressor(random_state=42, verbose=-1),
                    'params': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
                }
            if CATBOOST_INSTALLED:
                models['CatBoost'] = {
                    'model': CatBoostRegressor(random_state=42, verbose=0),
                    'params': {'model__iterations': [50, 100], 'model__learning_rate': [0.01, 0.1], 'model__depth': [4, 6]}
                }
            
        return models

# ==========================================
# 3. CORE LOGIC: Training & Evaluation Engine
# ==========================================

class AutoMLEngine:
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col
        self.detector = TaskDetector(data, target_col)
        self.task_info = self.detector.detect()
        self.builder = PipelineBuilder(self.task_info)
        self.results = {}
        self.best_model_name = None
        self.best_pipeline = None
        self.X_test = None
        self.y_test = None

    def run(self):
        # 1. Split Data
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]
        
        stratify_strategy = None
        
        if self.task_info['type'] == 'Classification':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            y = pd.Series(np.array(y_encoded), name=self.target_col)
            
            min_class_samples = y.value_counts().min()
            if min_class_samples >= 2:
                stratify_strategy = y 
            else:
                stratify_strategy = None
                if self.task_info['is_imbalanced']:
                    st.warning("⚠️ Some classes have too few samples (<2). Stratified splitting disabled.")

            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_strategy)
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
                stratify_strategy = None
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Store test data for later access
        self.X_test = X_test
        self.y_test = y_test

        preprocessor = self.builder.get_preprocessor(X_train)
        model_candidates = self.builder.get_models()
        
        progress_bar = st.progress(0)
        idx = 0
        total = len(model_candidates)

        for name, config in model_candidates.items():
            steps = []
            steps.append(('preprocessor', preprocessor))

            min_samples = y_train.value_counts().min()
            use_smote = (self.task_info['is_imbalanced'] and 
                         IMBLEARN_INSTALLED and 
                         min_samples >= 6)

            if use_smote:
                steps.append(('smote', SMOTE(random_state=42)))
                pipeline_cls = ImbPipeline
            else:
                pipeline_cls = Pipeline

            steps.append(('model', config['model']))
            full_pipeline = pipeline_cls(steps)

            if self.task_info['type'] == 'Classification' and stratify_strategy is not None:
                cv = StratifiedKFold(n_splits=3) 
            else:
                cv = KFold(n_splits=3)
            
            if self.task_info['type'] == 'Classification':
                scoring = 'f1_macro' if self.task_info['is_imbalanced'] else 'accuracy'
            else:
                scoring = 'r2'
            
            try:
                grid = GridSearchCV(full_pipeline, config['params'], cv=cv, scoring=scoring, n_jobs=-1)
                grid.fit(X_train, y_train)
                
                y_pred = grid.predict(X_test)
                metrics = {}
                
                if self.task_info['type'] == 'Classification':
                    metrics['Accuracy'] = accuracy_score(y_test, y_pred)
                    metrics['F1 Score'] = f1_score(y_test, y_pred, average='weighted')
                    if hasattr(grid, "predict_proba") and self.task_info['subtype'] == 'Binary':
                         try:
                             metrics['AUC-ROC'] = roc_auc_score(y_test, grid.predict_proba(X_test)[:, 1])
                         except:
                             pass
                else:
                    metrics['R2 Score'] = r2_score(y_test, y_pred)
                    metrics['RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
                    metrics['MAE'] = mean_absolute_error(y_test, y_pred)

                self.results[name] = {
                    'model': grid.best_estimator_, 
                    'params': grid.best_params_,
                    'metrics': metrics,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
            except Exception as e:
                # print(f"Error in {name}: {e}")
                continue
            
            idx += 1
            progress_bar.progress(idx / total)

        if not self.results:
            return None, None, None, None

        if self.task_info['type'] == 'Classification':
            primary_metric = 'F1 Score' if self.task_info['is_imbalanced'] else 'Accuracy'
            self.best_model_name = max(self.results, key=lambda k: self.results[k]['metrics'][primary_metric])
        else:
            primary_metric = 'R2 Score'
            self.best_model_name = max(self.results, key=lambda k: self.results[k]['metrics'][primary_metric])
            
        return self.results, self.best_model_name, X_test, y_test

# ==========================================
# 4. STREAMLIT UI IMPLEMENTATION
# ==========================================

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

@st.cache_resource
def train_models(_engine):
    return _engine.run()

def main():
    st.set_page_config(page_title="Advanced AutoML Engine", layout="wide")
    
    st.title("🚀 Advanced AutoML Engine")
    st.markdown("Automated Machine Learning for Classification & Regression.")
    
    st.sidebar.markdown("### 🛠️ Library Status")
    st.sidebar.caption(f"LightGBM: {'✅' if LGBM_INSTALLED else '❌'}")
    st.sidebar.caption(f"CatBoost: {'✅' if CATBOOST_INSTALLED else '❌'}")
    
    # --- Sidebar: Configuration ---
    st.sidebar.header("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.sidebar.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Select Target
        st.sidebar.header("2. Settings")
        target_col = st.sidebar.selectbox("Select Target Variable", df.columns)
        
        # --- TABS for better UX ---
        tab1, tab2 = st.tabs(["📊 Data Overview", "🤖 Model Training"])
        
        with tab1:
            st.subheader("Data Snapshot")
            st.dataframe(df.head())
            
            st.subheader("Statistics")
            st.write(df.describe())
            
            st.subheader("Target Distribution")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=target_col, kde=True, ax=ax)
            st.pyplot(fig)
            
            # Correlation Matrix (Numerical only)
            st.subheader("Correlation Matrix")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)
        
        with tab2:
            if st.button("⚡ Run Analysis"):
                with st.spinner("Training models... (This may take a moment)"):
                    engine = AutoMLEngine(df, target_col)
                    
                    # Unpack returned values including X_test and y_test
                    results, best_model_name, X_test, y_test = train_models(engine)
                    
                    if results is None:
                        st.error("Training Failed. Check dataset quality.")
                        st.stop()
                        
                    task_info = engine.task_info
                
                # --- RESULTS DISPLAY ---
                st.divider()
                st.subheader("🔍 Task Detection")
                c1, c2, c3 = st.columns(3)
                c1.metric("Type", task_info['type'])
                c2.metric("Sub-Type", task_info['subtype'])
                c3.metric("Imbalance", "Yes" if task_info['is_imbalanced'] else "No")

                st.subheader("🏆 Leaderboard")
                leaderboard_data = []
                for name, data in results.items():
                    row = {'Model': name}
                    row.update(data['metrics'])
                    leaderboard_data.append(row)
                st.dataframe(pd.DataFrame(leaderboard_data).style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
                
                # Best Model Analysis
                st.subheader(f"🥇 Best Model: {best_model_name}")
                best_res = results[best_model_name]
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("### 📊 Performance Metrics")
                    st.json(best_res['metrics'])
                    
                    model_buffer = io.BytesIO()
                    joblib.dump(best_res['model'], model_buffer)
                    b64 = base64.b64encode(model_buffer.getvalue()).decode()
                    href = f'<a href="data:file/pkl;base64,{b64}" download="best_model_{best_model_name}.pkl">📥 Download Trained Model (.pkl)</a>'
                    st.markdown(href, unsafe_allow_html=True)

                with col2:
                    if task_info['type'] == 'Classification':
                        st.markdown("**Confusion Matrix**")
                        fig, ax = plt.subplots()
                        
                        # FIX: Explicitly cast to integer to fix "continuous vs multiclass" error
                        y_true = best_res['y_test'].astype(int)
                        y_pred = best_res['y_pred'].astype(int)
                        
                        cm = confusion_matrix(y_true, y_pred)
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        st.pyplot(fig)
                    else:
                        st.markdown("**Actual vs Predicted**")
                        fig, ax = plt.subplots()
                        sns.scatterplot(x=best_res['y_test'], y=best_res['y_pred'], alpha=0.6)
                        min_val = min(min(best_res['y_test']), min(best_res['y_pred']))
                        max_val = max(max(best_res['y_test']), max(best_res['y_pred']))
                        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
                        st.pyplot(fig)
                
                # --- Feature Importance ---
                st.subheader("✨ Feature Importance (Permutation)")
                try:
                    # Use the explicit X_test and y_test from the cached run
                    if X_test is not None and y_test is not None:
                        result = permutation_importance(
                            best_res['model'], 
                            X_test, 
                            y_test, 
                            n_repeats=5, 
                            random_state=42
                        )
                        sorted_idx = result.importances_mean.argsort()[::-1][:10]
                        
                        fig, ax = plt.subplots()
                        plt.bar(range(len(sorted_idx)), result.importances_mean[sorted_idx], align="center")
                        feature_names = X_test.columns
                        plt.xticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
                        plt.xlabel("Feature")
                        plt.title("Permutation Importance")
                        st.pyplot(fig)
                    else:
                        st.warning("Test data unavailable for feature importance.")
                except Exception as e:
                    st.info(f"Feature importance could not be calculated: {e}")

    else:
        st.info("Awaiting file upload...")

if __name__ == "__main__":
    main()
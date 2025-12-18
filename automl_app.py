import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib
import base64

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, r2_score, mean_squared_error, 
    mean_absolute_error, confusion_matrix
)
from xgboost import XGBClassifier, XGBRegressor


try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_INSTALLED = True
except ImportError:
    st.warning("imbalanced-learn is not installed. SMOTE will be disabled. Run: pip install imbalanced-learn")
    IMBLEARN_INSTALLED = False

# task detection and data handling

class TaskDetector:
    def __init__(self, data, target_col):
        self.data = data
        self.target_col = target_col
        self.target_data = data[target_col]
    
    def detect(self):
        """
        Analyzes the target variable to determine problem type and characteristics.
        """
        unique_vals = self.target_data.nunique()
        total_rows = len(self.target_data)
        dtype = self.target_data.dtype
        
        info = {
            'type': 'Unknown',
            'subtype': None,
            'is_imbalanced': False,
            'imbalance_ratio': None
        }

        if unique_vals > 50 and (unique_vals / total_rows > 0.90):
             st.error(f"⚠️ **CRITICAL WARNING**: Target column '{self.target_col}' has {unique_vals} unique values (almost unique per row).")
             st.warning("It looks like an **ID** or **Date**. Classification models will fail (Accuracy ≈ 0%) because the classes in the Test set won't exist in the Training set. Please select a Category or Numerical column.")


        if (unique_vals <= 20) or (dtype == 'object') or (dtype == 'bool'):
            info['type'] = 'Classification'
            
            # Subtype: Binary vs Multiclass
            if unique_vals == 2:
                info['subtype'] = 'Binary'
            else:
                info['subtype'] = 'Multiclass'
            
            # Imbalance Detection
            value_counts = self.target_data.value_counts(normalize=True)
            min_class_ratio = value_counts.min()
            info['imbalance_ratio'] = min_class_ratio
            
            # If smallest class is less than 20%
            if min_class_ratio < 0.20: 
                info['is_imbalanced'] = True
        
        else:
            info['type'] = 'Regression'
            info['subtype'] = 'Continuous'
        
        return info
# pipeline abd preprocessing

class PipelineBuilder:
    def __init__(self, task_info):
        self.task_info = task_info
        
    def get_preprocessor(self, X):
        """Builds the column transformer for numeric and categorical data."""
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns
        
        # --- Advanced Imputation & Robust Scaling ---
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
            ]
        )
        return preprocessor

    def get_models(self):
        """Returns a dictionary of models + hyperparameter grids based on task."""
        models = {}
        
        if self.task_info['type'] == 'Classification':
            models['Logistic Regression'] = {
                'model': LogisticRegression(max_iter=1000, solver='liblinear'),
                'params': {'model__C': [0.1, 1, 10]}
            }
            models['Random Forest'] = {
                'model': RandomForestClassifier(random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10, 20]}
            }
            models['XGBoost'] = {
                'model': XGBClassifier(eval_metric='logloss', random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
            }
        else:
            models['Ridge Regression'] = {
                'model': Ridge(),
                'params': {'model__alpha': [0.1, 1.0, 10.0]}
            }
            models['Random Forest'] = {
                'model': RandomForestRegressor(random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__max_depth': [None, 10]}
            }
            models['XGBoost'] = {
                'model': XGBRegressor(random_state=42),
                'params': {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
            }
            
        return models
#training and evaluation

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

    def run(self):
        # 1. Split Data
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]
        
        stratify_strategy = None
        
        if self.task_info['type'] == 'Classification':
            

            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            y = pd.Series(np.array(y_encoded), name=self.target_col)
            
            # satisfaction check (imbalances)
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
                # Fallback to random split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)
                stratify_strategy = None
        

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 2. Get Components
        preprocessor = self.builder.get_preprocessor(X_train)
        model_candidates = self.builder.get_models()
        
        # 3. Training Loop with GridSearch
        progress_bar = st.progress(0)
        idx = 0
        total = len(model_candidates)

        for name, config in model_candidates.items():
            

            steps = []
            steps.append(('preprocessor', preprocessor))

            # SMOTE CHECK:
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

            # Grid Search Setup
            # Only use StratifiedKFold if we successfully stratified earlier
            if self.task_info['type'] == 'Classification' and stratify_strategy is not None:
                cv = StratifiedKFold(n_splits=3) 
            else:
                cv = KFold(n_splits=3)
            
            # Metric Selection
            if self.task_info['type'] == 'Classification':
                scoring = 'f1_macro' if self.task_info['is_imbalanced'] else 'accuracy'
            else:
                scoring = 'r2'
            try:
                grid = GridSearchCV(full_pipeline, config['params'], cv=cv, scoring=scoring, n_jobs=-1)
                grid.fit(X_train, y_train)
                
                # Evaluation
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
                # Log the error to UI but continue
                st.toast(f"⚠️ Model {name} failed: Data too sparse or incompatible. Skipping.", icon="⚠️")
                continue
            
            idx += 1
            progress_bar.progress(idx / total)

        # 4. Select Best Model
        if not self.results:
            st.error("❌ All models failed to train. Your dataset might be too small or contains incompatible data.")
            return None, None

        if self.task_info['type'] == 'Classification':
            primary_metric = 'F1 Score' if self.task_info['is_imbalanced'] else 'Accuracy'
            self.best_model_name = max(self.results, key=lambda k: self.results[k]['metrics'][primary_metric])
        else:
            primary_metric = 'R2 Score'
            self.best_model_name = max(self.results, key=lambda k: self.results[k]['metrics'][primary_metric])
            
        self.best_pipeline = self.results[self.best_model_name]['model']
        
        return self.results, self.best_model_name


@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def main():
    st.set_page_config(page_title="Advanced AutoML Engine", layout="wide")
    
    st.title("🚀 Advanced AutoML Engine")
    st.markdown("""
    **Features:** Automated Task Detection, Robust Preprocessing (KNN Imputation + Robust Scaling), 
    and Smart Model Selection.
    """)
    
    # --- Sidebar: Configuration ---
    st.sidebar.header("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.sidebar.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Select Target
        st.sidebar.header("2. Settings")
        target_col = st.sidebar.selectbox("Select Target Variable", df.columns)
        
        if st.sidebar.button("⚡ Run Analysis"):
            
            # --- RUNNING ENGINE ---
            with st.spinner("Analyzing dataset, running imputation, and training models..."):
                engine = AutoMLEngine(df, target_col)
                result_tuple = engine.run()
                
                if result_tuple[0] is None:
                    st.stop() # Stop execution if all models failed
                    
                results, best_model_name = result_tuple
                task_info = engine.task_info
            
            st.divider()
            
            # 1. Task Detection Report
            st.subheader("🔍 Task Detection Report")
            c1, c2, c3 = st.columns(3)
            c1.metric("Problem Type", task_info['type'])
            c2.metric("Sub-Type", task_info['subtype'])
            c3.metric("Imbalance Detected?", "Yes" if task_info['is_imbalanced'] else "No", 
                      delta_color="inverse" if task_info['is_imbalanced'] else "normal")

            # 2. Leaderboard
            st.subheader("🏆 Model Leaderboard")
            
            leaderboard_data = []
            for name, data in results.items():
                row = {'Model': name}
                row.update(data['metrics'])
                leaderboard_data.append(row)
            
            leaderboard_df = pd.DataFrame(leaderboard_data)
            st.dataframe(leaderboard_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            
            # 3. Best Model Analysis
            st.subheader(f"🥇 Best Model: {best_model_name}")
            best_res = results[best_model_name]
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📊 Performance Metrics")
                st.json(best_res['metrics'])
                
                # Download Model Button
                model_buffer = io.BytesIO()
                joblib.dump(best_res['model'], model_buffer)
                b64 = base64.b64encode(model_buffer.getvalue()).decode()
                href = f'<a href="data:file/pkl;base64,{b64}" download="best_model_{best_model_name}.pkl">📥 Download Trained Model (.pkl)</a>'
                st.markdown(href, unsafe_allow_html=True)

            with col2:
                # visual
                if task_info['type'] == 'Classification':
                    # Standard Confusion Matrix
                    st.markdown("### Confusion Matrix")
                    fig, ax = plt.subplots()
                    cm = confusion_matrix(best_res['y_test'], best_res['y_pred'])
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    plt.xlabel("Predicted Label")
                    plt.ylabel("True Label")
                    st.pyplot(fig)
                    
                else:
                    # Regression Plot
                    st.markdown("### Actual vs Predicted")
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=best_res['y_test'], y=best_res['y_pred'], alpha=0.6)
                    
                    min_val = min(min(best_res['y_test']), min(best_res['y_pred']))
                    max_val = max(max(best_res['y_test']), max(best_res['y_pred']))
                    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
                    
                    plt.xlabel("Actual")
                    plt.ylabel("Predicted")
                    st.pyplot(fig)
                    

    else:
        st.info("Awaiting file upload...")

if __name__ == "__main__":
    main()
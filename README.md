# 🚀 Advanced AutoML Engine

A robust, no-code Automated Machine Learning (AutoML) tool built with **Python** and **Streamlit**. This application democratizes predictive modeling by allowing users to upload a raw dataset and automatically receive a fully trained, optimized, and downloadable machine learning model without writing a single line of code.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

* **🧠 Intelligent Task Detection:** Automatically analyzes the target variable to determine if the problem is **Classification** (Binary/Multiclass) or **Regression**.
    * Detects **Imbalanced Datasets** and applies SMOTE where appropriate.
    * **Guardrails:** Detects and warns users against selecting ID or Date columns to prevent model failure.
* **🧹 Robust Preprocessing Pipeline:**
    * **Imputation:** Fills missing values using **K-Nearest Neighbors (KNN)**.
    * **Scaling:** Applies **RobustScaler** to handle outliers effectively.
    * **Encoding:** Automatically handles categorical variables using One-Hot Encoding.
* **🏎️ Massive Model Library:** Trains and compares 10+ algorithms, including:
    * **Boosting:** XGBoost, LightGBM, CatBoost, AdaBoost.
    * **Ensembles:** Random Forest, Extra Trees.
    * **Classic Models:** SVM, KNN, Logistic/Ridge Regression, Naive Bayes.
* **🛡️ Fault-Tolerant Engineering:**
    * Uses caching (`@st.cache_resource`) to prevent unnecessary retraining.
    * Implements **Try-Except** blocks to handle data sparsity or library failures gracefully.
* **📊 Interactive Dashboards:**
    * **EDA Tab:** Visualize distributions and correlations before training.
    * **Visualizations:** Dynamic Confusion Matrices and Actual vs. Predicted plots.
    * **Explainability:** Permutation Feature Importance plots to understand model drivers.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mbenbou/automl-engine.git](https://github.com/mbenbou/automl-engine.git)
    cd automl-engine
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

1.  **Run the App:**
    ```bash
    streamlit run automl_app.py
    ```

2.  **Workflow:**
    * **Upload:** Drag and drop your CSV or Excel file.
    * **Select Target:** Choose the column you want to predict.
    * **Analyze:** Use the "Data Overview" tab to check your data.
    * **Train:** Click **⚡ Run Analysis**.
    * **Download:** Save the best model (`.pkl`) for future use.

---

## 📦 Models Supported

| Algorithm | Classification | Regression | Library |
| :--- | :---: | :---: | :--- |
| **Random Forest** | ✅ | ✅ | Scikit-Learn |
| **Extra Trees** | ✅ | ✅ | Scikit-Learn |
| **XGBoost** | ✅ | ✅ | XGBoost |
| **LightGBM** | ✅ | ✅ | LightGBM |
| **CatBoost** | ✅ | ✅ | CatBoost |
| **AdaBoost** | ✅ | ✅ | Scikit-Learn |
| **SVM / SVR** | ✅ | ✅ | Scikit-Learn |
| **KNN** | ✅ | ✅ | Scikit-Learn |
| **Logistic Regression** | ✅ | ❌ | Scikit-Learn |
| **Ridge Regression** | ❌ | ✅ | Scikit-Learn |
| **Naive Bayes** | ✅ | ❌ | Scikit-Learn |

---

## 🧠 Technical Architecture

The engine uses a modular Object-Oriented design:

1.  **`TaskDetector`**: Analyzes metadata (cardinality, dtype) to route the problem to the correct logic.
2.  **`PipelineBuilder`**: Dynamically assembles Scikit-Learn pipelines. It forces Pandas output format to ensure feature names are preserved for advanced models like LightGBM.
3.  **`AutoMLEngine`**: Manages the training loop. It handles Stratified vs Random splitting based on class balance and catches errors on individual models to ensure the application never crashes completely.
<img width="1910" height="922" alt="image" src="https://github.com/user-attachments/assets/eb78353e-62e0-4878-b1bf-4a178ff3be90" />
<img width="1871" height="923" alt="image" src="https://github.com/user-attachments/assets/5e8a7a4b-e8ad-4fb1-8e1f-89cf971516bd" />
---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

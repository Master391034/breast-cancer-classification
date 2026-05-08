# Breast Cancer Classification

**Predicting malignant vs. benign tumors from cell nuclei features.**

This project uses the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) to build a production‑ready classifier that minimises false negatives – critical in a medical screening context.

---

## Dataset

The dataset consists of **569 samples** (357 benign, 212 malignant) with **30 numeric features** computed from digitised fine‑needle aspirate images. For each cell nucleus, ten real‑valued characteristics are measured:

- radius (mean of distances from centre to perimeter)
- texture (standard deviation of grey‑scale values)
- perimeter
- area
- smoothness (local variation in radius)
- compactness (perimeter² / area − 1)
- concavity (severity of concave contour portions)
- concave points (number of concave portions)
- symmetry
- fractal dimension (“coastline approximation” − 1)

For every image the **mean**, **standard error** and **worst** (average of the three largest values) are recorded, giving 30 features in total.

## Methodology

1. **Exploratory Data Analysis** – class distribution, feature correlation heatmaps (mean / SE / worst groups).
2. **Data Preprocessing** – StandardScaler for linear & distance‑based models; no scaling for tree‑based ensembles.
3. **Model Benchmarking** – 7 models evaluated via 5‑fold stratified cross‑validation (accuracy, F1, ROC‑AUC):
   - Logistic Regression
   - K‑Nearest Neighbours
   - Support Vector Machine (RBF)
   - Decision Tree
   - Random Forest
   - XGBoost
   - CatBoost
4. **Hyperparameter Optimisation** – Bayesian search with [Optuna](https://optuna.org/) for XGBoost (learning rate, tree depth, subsampling, regularisation).
5. **Interpretability** – [`SHAP`](https://github.com/slundberg/shap) analysis to identify the most influential features.
6. **Threshold Analysis** – examined the trade‑off between false negatives (missed cancers) and false positives.
7. **Final Model** – XGBoost trained on raw (unscaled) data, saved with `joblib`.

---

## Key Results

| Model               | F1 (test) | ROC‑AUC (test) |
|---------------------|-----------|----------------|
| **XGBoost (tuned)** | **0.992** | **0.995**       |
| CatBoost            | 0.985     | 0.994           |
| Random Forest       | 0.970     | 0.981           |
| Logistic Regression | 0.960     | 0.998           |

**XGBoost** was selected for production because of its highest F1 score and stable cross‑validation performance.

**SHAP insights:** the top predictors are `concave points_mean`, `texture_worst`, `perimeter_worst`, `area_se`, and `concavity_worst`. These align with medical knowledge – malignant cells tend to have more irregular contours and larger nuclei.

**Outlier analysis:** the single false negative (1 out of 64 malignant cases) has a predicted probability close to zero, behaving as a strong outlier. Attempting to capture it by lowering the threshold would drastically increase false positives. We keep the default **0.5** threshold, achieving **99% recall** while maintaining robustness.

## Repository Structure
├── breast_cancer_ml.ipynb # Full analysis notebook
├── xgboost_breast_cancer_model.pkl # Trained XGBoost model (raw features)
├── data.csv # Wisconsin Diagnostic Dataset
├── requirements.txt # Python dependencies
└── README.md # This file

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

**SHAP insights:** 
SHAP analysis showed that the model relies on the same nuclear morphology features that a pathologist uses when examining FNA biopsy samples under a microscope. Key predictors of malignancy:

- **`concave points_mean` and `concavity_worst`** — number and depth of indentations on the nuclear contour. Nuclear area, perimeter, diameter, compactness, and concave points were found to be statistically significant (P < 0.05) parameters in differentiating benign and malignant breast aspirates. Concave points are the most influential feature in classifying the breast mass as benign or malignant. [Significance of nuclear morphometry in benign and malignant breast aspirates (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3678677/)

- **`perimeter_worst` and `area_worst`** — enlarged and irregularly shaped nuclei. Mean nuclear area and perimeter were useful in differentiating benign and malignant breast aspirates; ductal carcinoma cells showed higher values for nuclear area, perimeter, diameter, compactness, and concave points when compared to fibroadenomas and fibrocystic disease. [Significance of nuclear morphometry in benign and malignant breast aspirates (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3678677/)

- **`area_se`** — high variability of nuclear sizes within the sample (nuclear pleomorphism). Nuclear pleomorphism has a significant prognostic impact on distant recurrence-free survival and overall survival (DRFS p=0.0002, OS p=0.00001). [Nuclear pleomorphism, a strong prognostic factor in breast cancer (Springer)](https://link.springer.com/article/10.1007/BF01834640)

- **`texture_worst` and `texture_mean`** — heterogeneity of chromatin distribution within the nucleus. One component of tumor grading is the nuclear pleomorphism score — the extent of abnormalities in the overall appearance of tumor nuclei — subjectively classified from 1 to 3, where 1 most closely resembles normal breast epithelium and 3 shows the greatest abnormalities. [Deep learning for fully-automated nuclear pleomorphism scoring (npj Breast Cancer, Nature)](https://www.nature.com/articles/s41523-022-00488-w)

- **`radius_worst`** — a large radius of the "worst" nuclei indicates abnormal enlargement. radius-worst is an important factor affecting breast cancer occurrence, and its interaction with concave points-worst exists. [Breast cancer prediction modeling based on SHAP interpretability analysis and XGBoost (ResearchGate)](https://www.researchgate.net/publication/390880576_Breast_cancer_prediction_modeling_based_on_SHAP_interpretability_analysis_and_XGBoost_algorithm)

**Outlier analysis:** the single false negative (1 out of 64 malignant cases) has a predicted probability close to zero, behaving as a strong outlier. Attempting to capture it by lowering the threshold would drastically increase false positives. We keep the default **0.5** threshold, achieving **99% recall** while maintaining robustness.

## Repository Structure
├── breast_cancer_ml.ipynb # Full analysis notebook
├── xgboost_breast_cancer_model.pkl # Trained XGBoost model (raw features)
├── data.csv # Wisconsin Diagnostic Dataset
├── requirements.txt # Python dependencies
└── README.md # This file

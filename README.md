# 🏥 Medical Insurance Premium & Charges Prediction

A comprehensive Machine Learning pipeline and interactive Jupyter notebook for predicting individual medical insurance costs based on demographic and health indicators using the `insurance.csv` dataset.

---

## 📁 Repository Contents

| File | Description |
| :--- | :--- |
| **[`insurance.csv`](./insurance.csv)** | Raw medical insurance dataset containing 1,338 records and 7 demographic/health attributes. |
| **[`ML01(insurance)(1).ipynb`](./ML01(insurance)(1).ipynb)** | Complete end-to-end Jupyter Notebook covering EDA, data cleaning, feature engineering, statistical feature selection, multi-model regression benchmarking, and evaluation. |
| **[`README.md`](./README.md)** | Project documentation, dataset overview, pipeline architecture, and benchmark results. |

---

## 📊 Dataset Overview

The dataset contains records of **1,338 individuals** with attributes covering health, lifestyle, and demographic variables:

| Column Name | Data Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `age` | Integer | Age of primary beneficiary | 18 – 64 years |
| `sex` | Categorical | Insurance contractor gender | `female`, `male` |
| `bmi` | Float | Body mass index ($kg/m^2$), ideal is 18.5 to 24.9 | 15.96 – 53.13 |
| `children` | Integer | Number of children / dependents covered | 0 – 5 |
| `smoker` | Categorical | Smoking status of the individual | `yes`, `no` |
| `region` | Categorical | Residential area in the US | `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | Float | Individual medical costs billed by health insurance (**Target**) | $1,121.87 – $63,770.43 |

---

## 🔬 Pipeline & Methodology

### 1. Exploratory Data Analysis (EDA) & Cleaning
- **Missing Values & Deduplication**: Verified data integrity, handled duplicates, and validated data types.
- **Distribution Analysis**: Analyzed target variable (`charges`) skewness and examined distributions across demographic factors.
- **Clinical Risk Multipliers**: Analyzed the strong synergistic effect of smoking and obesity on insurance charges:
  - **Non-Smoker, Non-Obese**: Mean charges $\approx$ **$7,977**
  - **Non-Smoker, Obese**: Mean charges $\approx$ **$8,843**
  - **Smoker, Non-Obese**: Mean charges $\approx$ **$21,363**
  - **Smoker, Obese**: Mean charges $\approx$ **$41,558** *(5.2x higher than baseline)*

### 2. Feature Engineering & Selection
- **Type-Safe Encoding**: Converted categorical variables (`sex`, `smoker`, `region`) using one-hot encoding without precision loss.
- **Domain Interaction Features**:
  - `smoker_obese_interaction`: Binary flag capturing compound risk (`is_smoker * is_obese`).
  - `smoker_bmi`: Continuous interaction term (`is_smoker * bmi`).
  - `age_squared`: Non-linear age trajectory term ($age^2$).
- **Statistical Feature Selection**: Applied continuous regression diagnostics using **ANOVA F-statistic (`f_regression`)**, **Mutual Information Regression (`mutual_info_regression`)**, and Variance Inflation Factor (VIF) checks.
- **Leak-Free Preprocessing**: Applied `train_test_split` (80/20 partition) before feature scaling (`StandardScaler`) wrapped inside pipelines.

---

## 📈 Model Benchmarks & Results

Benchmarked across multiple regression models with **5-Fold Cross-Validation**:

| Model | 5-Fold CV $R^2$ | Test $R^2$ | Test RMSE ($) | Test MAE ($) |
| :--- | :---: | :---: | :---: | :---: |
| 🥇 **Gradient Boosting Regressor** | **0.8530** | **0.8998** | **$4,039.63** | **$2,395.74** |
| 🥈 **Random Forest Regressor** | 0.8540 | **0.8939** | $4,156.45 | $2,442.27 |
| 🥉 **HistGradientBoosting** | 0.8402 | **0.8872** | $4,285.80 | $2,640.40 |
| 🔹 **Lasso Regression ($\alpha=10$)** | 0.8514 | **0.8805** | $4,410.37 | $2,777.62 |
| 🔹 **Linear Regression** | 0.8354 | **0.8805** | $4,410.74 | $2,781.04 |
| 🔹 **Ridge Regression ($\alpha=1$)** | 0.8356 | **0.8805** | $4,40.60 | $2,780.05 |

> **Key Finding**: Non-linear tree ensembles (Gradient Boosting & Random Forest) significantly outperform linear baselines, boosting $R^2$ to **~0.90** and reducing prediction error by over **$2,000** per individual.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/p4nd3y/Insurance_Model_Data.git
   cd Insurance_Model_Data
   ```

2. Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook "ML01(insurance)(1).ipynb"
   ```

3. Run the cells sequentially to reproduce the analysis, feature transformations, visualizations, and model evaluations.

---

## 📄 License
This project is open-source and available for educational and analytical purposes.

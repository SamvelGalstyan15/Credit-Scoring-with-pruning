# Credit Default Prediction & Overfitting Prevention via Tree Pruning (XGBoost)

A machine learning model designed to assess credit risk and predict the probability of borrower default within the next 2 years based on the **Give Me Some Credit** dataset (150,000 records). The core focus of this project is exploring **tree pruning** techniques to prevent overfitting on noisy financial data.

## 📌 Project Objective
Build an overfitting-resistant classification model on imbalanced financial data (~6.5% default rate), tune tree pruning hyperparameters, and calibrate the decision threshold to minimize financial losses for the lender.

## 📊 Data & Preprocessing
* **Source:** Kaggle — *Give Me Some Credit* (`cs-training.csv`, 150,000 rows, 10 features).
* **Target Variable:** `SeriousDlqin2yrs` (binary flag indicating a delinquency of 90+ days).
* **Feature Processing:** 
  * Dropped the redundant technical index column `Unnamed: 0`.
  * Missing values in `MonthlyIncome` and `NumberOfDependents` were handled natively by the gradient boosting algorithm.
  * Split the dataset into training (80%) and validation (20% — 30,000 clients) sets (`random_state=42`).

## ⚙️ Model Architecture & Tree Pruning
During experimentation, a single Decision Tree with post-pruning (`Cost-Complexity Pruning` via `ccp_alpha`, ROC-AUC `0.8415`) was upgraded to **`XGBClassifier`**, resulting in a substantial performance boost.

* **Algorithm:** `XGBClassifier` (`n_estimators=200`, `learning_rate=0.05`).
* **Branch Pruning:** Tuned the minimum loss reduction parameter required to make a further partition on a leaf node (`gamma`: `[0.1, 0.3, 0.8]`) using `GridSearchCV` (`cv=3`, optimizing for `roc_auc`).
* **Threshold Calibration:** Instead of using the default `0.5` cutoff, **Youden's J statistic** ($\max(\text{TPR} - \text{FPR})$) was applied to the ROC curve to adapt the model to severe class imbalance.

## 📈 Results & Metrics (Validation Set — 30,000 Clients)

| Metric | Score | Description |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.8641** | Strong capability to rank borrowers by default risk |
| **Recall (Class 1 — Defaulters)** | **0.78** | Successfully identifies **78%** of all actual defaulters (`1,532` out of `1,956`) |
| **Recall (Class 0 — Non-Defaulters)** | **0.79** | Correctly approves **79%** of reliable borrowers (`22,186` out of `28,044`) |
| **Precision (Class 1)** | **0.21** | Expected trade-off for high risk coverage on an imbalanced dataset |

### ROC Curve & Optimal Threshold
The chart below illustrates the trade-off between True Positive Rate (`TPR`) and False Positive Rate (`FPR`). The red dot marks the optimal decision threshold (`FPR ≈ 0.21`, `TPR ≈ 0.78`) selected via Youden's index.

<img width="608" height="476" alt="image (1)" src="https://github.com/user-attachments/assets/5384ada1-8fbf-4e32-88c9-1768b63dc366" />




### Confusion Matrix (with Optimal Youden's Threshold)
* **True Negative (Correctly Approved):** 22,186
* **False Positive (False Alarm / Rejected):** 5,858
* **False Negative (Missed Default):** 424
* **True Positive (Caught Default):** 1,532

## 🛠 Tech Stack
* **Python 3**
* **Pandas / NumPy** — data manipulation and numerical operations
* **Scikit-learn** — train-validation splitting, `GridSearchCV`, and evaluation metrics (`roc_auc_score`, `roc_curve`, `classification_report`, `confusion_matrix`)
* **XGBoost** — gradient boosting model training, branch pruning (`gamma`), and model serialization (`final_model.json`)
* **Matplotlib** — visualization of the ROC curve and optimal threshold point

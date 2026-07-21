[README_loan_default.md](https://github.com/user-attachments/files/30235252/README_loan_default.md)
# Loan Default Prediction

## Business Problem

Lenders (banks, microfinance institutions, etc.) take on credit risk every time they approve a loan — some borrowers repay on time, while others default or struggle to repay, which reduces profitability and increases losses.

**Business question:** which factors are most important in predicting the likelihood of loan default, and can a model reliably flag high-risk borrowers before a loan is approved?

**Dataset source:** [Loan Default Prediction (Kaggle)](https://www.kaggle.com/datasets/nikhil1e9/loan-default)

## Dataset

- **Size:** 255,347 rows, 18 columns (17 features after dropping the ID column)
- **Target:** `Default` (binary) — **highly imbalanced**: 225,694 non-default (~88.4%) vs. 29,653 default (~11.6%)
- **Dropped columns:** `LoanID` (non-predictive identifier)
- **No missing values** in the dataset
- **Feature types:**
  - **Numeric:** Age, Income, Loan Amount, Credit Score, Months Employed, Number of Credit Lines, Interest Rate, Loan Term, DTI (Debt-to-Income) Ratio
  - **Categorical:** Education, Employment Type, Marital Status, Has Mortgage, Has Dependents, Loan Purpose, Has Co-Signer

## Methodology

1. **Preprocessing pipeline** (via `ColumnTransformer`):
   - Numeric features: median imputation + standard scaling
   - Categorical features: most-frequent imputation + one-hot encoding
2. **Train/test split:** 80/20, stratified on `Default` to preserve the ~11.6% minority class rate in both sets
3. **Models trained** (all with class-imbalance handling):
   - **Logistic Regression** — `class_weight="balanced"`
   - **Random Forest** — 300 trees, `class_weight="balanced"`
   - **XGBoost** — 500 trees, depth 5, learning rate 0.05, `scale_pos_weight` set to the class ratio
4. **Evaluation metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, ROC curves
5. **Feature importance:** Permutation importance computed on the fitted Logistic Regression model

## Results

Standardized comparison (all models scored at the default 0.5 probability threshold):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.68 | 0.22 | **0.70** | 0.33 | 0.75 |
| Random Forest | **0.89** | **0.57** | 0.06 | 0.10 | 0.75 |
| XGBoost | 0.71 | 0.23 | 0.67 | **0.35** | **0.76** |

**Note on Random Forest:** at the default 0.5 threshold, Random Forest catches almost none of the actual defaults (recall = 0.06) despite its high overall accuracy — a classic symptom of accuracy being a misleading metric on an imbalanced dataset (predicting "no default" for almost everyone still scores ~88% accuracy). When the notebook instead used a lowered decision threshold (0.25) for Random Forest, recall improved substantially to 0.47, at the cost of more false positives — illustrating how much threshold choice matters for this kind of problem.

## Key Findings (Permutation Importance — Logistic Regression)

Ranked by importance, the strongest predictors of default were:

1. **Age** — by far the most important feature
2. **Interest Rate**
3. **Months Employed**
4. **Loan Amount**
5. **Income**

Categorical factors (Employment Type, Education, Has Co-Signer, etc.) and financial ratios like Credit Score and DTI Ratio had comparatively minor influence in the linear model — Loan Term and Marital Status were essentially uninformative.

## Model Trade-offs & Recommendation

- **Logistic Regression:** Highest recall (0.70) — catches most true defaulters, but low precision (0.22) means many false alarms. Simple and interpretable.
- **Random Forest:** Highest raw accuracy (0.89) and precision (0.57), but very poor recall (0.06) at default threshold — it plays it safe by rarely predicting default, which misses most actual defaulters.
- **XGBoost:** The best balance of precision, recall, and F1 (0.35), and the highest ROC-AUC (0.76) — a reasonable middle ground between the other two.

**Recommendation:** since missing an actual defaulter (false negative) is typically far costlier to a lender than incorrectly flagging a safe borrower (false positive), **recall on the default class is the priority metric**. On that basis, **Logistic Regression is the recommended model** for this use case despite its lower overall accuracy, because it identifies the largest share of true defaults. XGBoost is a strong alternative if a more balanced precision/recall trade-off is preferred, and Random Forest's threshold could be tuned down (as demonstrated) if its precision/interpretability advantages are wanted while improving recall.

## Caveats

- The dataset is a single Kaggle source; performance should be validated on real institutional lending data before any production use.
- Precision remains low (~0.22–0.35) across all models, meaning a meaningful share of flagged loans would not actually default — any deployment would need a business-defined cost trade-off between missed defaults and false alarms, and further threshold tuning.

## Files

- `Loan_default.csv` — input dataset (loaded from Google Drive in the notebook)
- Notebook: data loading → preprocessing → model training/evaluation (Logistic Regression, Random Forest, XGBoost) → permutation feature importance → model comparison

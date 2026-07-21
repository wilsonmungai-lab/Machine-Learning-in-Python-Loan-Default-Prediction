# Machine-Learning-in-Python-Loan-Default-Prediction
Which factors are most important when predicting the likelihood of loan default? This project seeks to create a Loan Default Prediction Algorithm 
# EXPLANATION

# Logistic Regression (Baseline)
Strengths High recall (0.70) → catches most of the minority class (1’s).

# Simple and interpretable.

# Weaknesses:

Very low precision (0.22) → lots of false positives.

Overall accuracy lower since it misclassifies many majority cases.

NB Best when the cost of missing a positive (false negative) is very high, e.g., loan default detection.

Random Forest
Strengths

Highest accuracy (0.89) and ROC-AUC (0.75).

Very strong at classifying majority class (0’s).

Weaknesses:

Low recall for minority class (0.06).

Identifies fewer actual positives (misses many).

Best when minimizing overall error is most important, but not ideal if recall on minority class is critical.

3. XGBoost

Strengths:

Balanced performance: higher recall (0.67) than RF, better F1-score (0.35) than both others.

Precision better than Logistic Regression, though lower than RF. ROC-AUC slightly bigger than the other two models. Weaknesses:

Accuracy lower than Random Forest (but still good at 0.71).

Best compromise model if you want to balance catching more positives and keeping false positives reasonable.

✅ Recommendation

If finding as many positives as possible is the priority → Logistic Regression (highest recall).

If overall accuracy and majority class prediction matter most → Random Forest.

If you need a balanced trade-off between recall, precision, and F1 → XGBoost is the best fit.

In this case, since the cost of missing a true positive is greater than incorrectly classifying a True negative as a True positive, a banking institution would prefer to adopt the Logistic regression as the Loan Default Prediction Algorithm

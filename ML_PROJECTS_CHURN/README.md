Customer Churn Prediction with XGBoost

1. Project Overview
This project predicts whether a telecom customer will cancel their subscription (churn) using XGBoost — one of the most powerful and widely used algorithms in industry. The dataset contains 7,043 customers with 20 features covering demographics, services, and billing information.

Field         |	Details
Dataset	      | Telco Customer Churn (IBM)
Problem Type  |	Binary Classification
Target Column |	Churn (Yes=1, No=0)
Dataset Size  |	7,043 rows, 23 features after engineering
Class Balance |	73.5% No Churn / 26.5% Churn (imbalanced)
Algorithm     |	XGBoost Classifier
Final Accuracy|	75%

2. Why This Problem Is Hard
The dataset is imbalanced — 73.5% of customers do NOT churn. This means a model that predicts 'no churn' for everyone would score 73.5% accuracy while being completely useless. This is why accuracy alone is not enough — recall and F1 score matter more here.

The Real Business Question
It is far more expensive to lose a customer than to incorrectly flag a loyal one. Therefore the model prioritises high recall on churners — catching as many real churners as possible — even if it means some false alarms.

False Negative (Miss a churner)         |	False Positive (Flag loyal customer)
Customer cancels — revenue lost forever |	Offer discount unnecessarily — small cost

3. Feature Engineering
Three new features were created from existing data to give the model more predictive power:

Monthly_Tenure_Ratio
Monthly charges divided by tenure plus one. Captures whether a customer is paying a lot relative to how long they have been with the company. New expensive customers are high risk.
df['Monthly_Tenure_Ratio'] = df['MonthlyCharges'] / (df['tenure'] + 1)

High_Risk_Combo
Binary flag for customers on Fiber optic internet with a month-to-month contract. This combination historically has the highest churn rate — expensive service with no commitment.
df['High_Risk_Combo'] = ((df['InternetService'] == 'Fiber optic') &
                         (df['Contract'] == 'Month-to-month')).astype(int)

Total_Addons
Count of how many additional services a customer subscribes to (security, backup, streaming etc). Customers with more addons are more embedded in the service and less likely to leave.
df['Total_Addons'] = df[addons].apply(lambda x: x == 'Yes').sum(axis=1)

4. Data Preparation
•Dropped customerID — unique identifier, no predictive value
•Converted TotalCharges from string to numeric — 11 hidden nulls discovered
•Filled 11 null TotalCharges values with median
•Encoded binary text columns manually (Yes/No, Male/Female) to 0/1
•Applied pd.get_dummies() for remaining categorical columns
•Used stratify=y in train_test_split to preserve class balance in both sets

Why stratify=y Matters
Without stratify, the random split might put most churners in training and very few in test — making evaluation unreliable. Stratify guarantees the same 73.5/26.5 ratio in both train and test sets.

5. Model — XGBoost
XGBoost (Extreme Gradient Boosting) builds trees sequentially — each new tree learns from the mistakes of all previous trees. Unlike Random Forest which builds trees independently and averages them, XGBoost focuses each new tree on the hardest examples.

Hyperparameter        |	Value and Meaning
n_estimators          |1000 — number of trees (stopped early by early_stopping_rounds)
max_depth	          | 3 — shallow trees, controls overfitting
learning_rate         |	0.05 — small steps, more careful learning
scale_pos_weight      |	2.77 — tells model to weight churners more heavily (class imbalance fix)
early_stopping_rounds |	10 — stops training if no improvement for 10 rounds
eval_metric	          | logloss — evaluation metric used during training

What scale_pos_weight Does
The dataset has 2.77 times more non-churners than churners. Setting scale_pos_weight=2.77 tells XGBoost to treat each churner as if it were 2.77 customers — this compensates for the imbalance and prevents the model from ignoring the minority class.

What early_stopping_rounds Does
Instead of training all 1000 trees blindly, the model watches performance on the test set after each tree. If 10 consecutive trees produce no improvement, training stops automatically. This prevents overfitting and saves computation time.

6. Hyperparameter Tuning Results
Trees |	LR   | SPW  |	Accuracy
100	  | 0.1  |  1   | 0.8013 BEST
100	  | 0.05 |	4   | 0.7175
100	  | 0.05 |	2.77| 0.7495

7. Final Model Results
Class        |	Precision  | Recall
0 — No Churn |	0.91       |	0.73
1 — Churn    |	0.51       |	0.80 — catches 80% of churners

Reading The Confusion Matrix
[[753  282]  — 753 correctly predicted no churn, 282 false alarms
 [75   299]] — only 75 missed churners, 299 correctly caught

The model catches 299 out of 374 actual churners (80% recall). Only 75 churners slipped through undetected. For a business this means 80% of at-risk customers can be targeted with retention offers before they leave.



8. Key Concepts Learned

XGBoost vs Random Forest

Random Forest |	XGBoost
Builds trees independently | Builds trees sequentially
Trees vote equally | Each tree fixes previous mistakes
No learning rate needed | Learning rate controls step size
Good baseline model | Usually more accurate, preferred in industry

Class Imbalance
When one class has far more samples than another (73.5% vs 26.5% here), standard models learn to always predict the majority class. Solutions include scale_pos_weight in XGBoost, stratified splitting, SMOTE oversampling, and choosing recall over accuracy as the primary metric.

verbose=False
During XGBoost training with 1000 trees, the model prints a performance score after every single tree by default. verbose=False silences this output. Without it the terminal floods with 1000 lines of logs.

Early Stopping
Instead of always training all n_estimators trees, early stopping monitors performance on a validation set after each tree. If performance does not improve for a set number of rounds (10 here), training stops automatically. This prevents overfitting and saves significant computation time on large datasets.


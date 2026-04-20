 Telco Customer Churn Prediction

 Project Objective
The goal of this project is to identify high-risk customers likely to churn from a telecommunications provider. By predicting churn accurately, the business can proactively engage at-risk customers with retention strategies (e.g., targeted discounts or contract renewals) to reduce revenue loss.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
* **Model:** RandomForestClassifier (Optimized for Recall)

Data Pipeline
1.  **Data Cleaning:** Handled "hidden" nulls in `TotalCharges` by forcing numeric conversion and median imputation.
2.  **Feature Engineering:** * Converted binary categories (Gender, Partner, etc.) to 0/1.
    * Applied One-Hot Encoding to multi-class features (Contract, InternetService).
    * Dropped redundant columns to avoid multicollinearity.
3.  **Handling Imbalance:** The dataset showed a **26.5% churn rate**. I utilized `class_weight='balanced'` within the Random Forest to ensure the model prioritized catching the minority (churn) class.

 Model Performance
Because missing a churner is more expensive than a false alarm, I prioritized **Recall** over Accuracy.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 77% |
| **Recall (Churn)** | **73%** |
| **Precision (Churn)** | 55% |
| **F1-Score** | 0.62 |

### Confusion Matrix

* **True Positives:** 273 (Customers we successfully caught before they left)
* **False Negatives:** 101 (Churners we missed—the "Danger Zone")

 Key Insights (Feature Importance)
Based on the Random Forest feature rankings, the top drivers of churn are:
1.  **Tenure:** New customers are significantly more likely to churn than long-term ones.
2.  **Contract Type:** Month-to-month contracts are the highest risk factor.
3.  **Charges:** High `MonthlyCharges` correlate strongly with customer exit.
4.  **Internet Service:** Fiber Optic users show a surprisingly higher churn rate, suggesting a need for service quality audits.

[Insert your Feature Importance chart here!]

Business Recommendations
* **Incentivize Long-term Contracts:** Offer small loyalty discounts to move Month-to-Month users to 1 or 2-year plans.
* **Fiber Optic Retention:** Investigate the Fiber Optic customer experience to identify if the churn is due to pricing or technical instability.
* **Early Intervention:** Focus retention marketing on customers within their first 6-12 months of service.



### Pro-Tips for the Repo:
1.  **The Image:** Save your Feature Importance plot as `feature_importance.png` in your folder, and then in the README, link it like this: `![Feature Importance](feature_importance.png)`.
2.  **How to Run:** Add a small section at the bottom:
    ```markdown
    ## 🏃 How to Run
    1. Clone the repo: `git clone [your-link]`
    2. Install requirements: `pip install -r requirements.txt`
    3. Run the script: `python customer_churn_prediction.py



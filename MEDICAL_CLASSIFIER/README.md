DIABETES RISK PREDICTOR
With SHAP Explainability — ML Engineering Project
Best Accuracy: 77.92% | Algorithm: Random Forest | Dataset: PIMA Indians (768 patients)

1. Project Overview
This project predicts whether a patient has diabetes using the PIMA Indians Diabetes dataset, one of the most well-known medical ML benchmarks. What makes this project different from a standard classifier is the addition of SHAP (SHapley Additive exPlanations), which explains every individual prediction in plain medical terms.

Field        |	Details
Dataset      |	PIMA Indians Diabetes Dataset
Source       |	github.com/jbrownlee/Datasets
Patients     |	768 total (500 healthy, 268 diabetic)
Features     |	8 medical measurements
Problem Type |	Binary Classification (Diabetic vs Healthy)
Best Model   |	Random Forest — 77.92% Accuracy
Key Feature  |	SHAP Waterfall plots — explains every individual prediction

2. The Data Problem — Hidden Zeros
This dataset has a trap that catches most beginners (it caught me). Several medical columns contain the value 0, but these are not real measurements, they are missing values recorded as zero.

Column        |	Why 0 is Impossible
BMI           |	A living person cannot have zero body mass
Glucose       |	Zero blood sugar means the patient would be in a coma
BloodPressure |	Zero blood pressure means the heart has stopped
Insulin       |	Cannot be zero in a living patient
SkinThickness |	Zero thickness is anatomically impossible

The Fix — Median Imputation
We replace all impossible zeros with NaN (empty) then fill them with the median value of that column. Median is used instead of average because extreme outliers can pull the average in one direction, whereas the median is always the middle value regardless of outliers.

cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
for col in cols_to_fix:
    df[col] = df[col].fillna(df[col].median())

3. Why Scaling Matters
The 8 features in this dataset exist on completely different numerical scales:

Feature          |	Range         |	Problem Without Scaling
Insulin          |	14 to 846     | Model thinks insulin is 350x more important than DiabetesPedigree
Age	             |  21 to 81      |	Reasonable range
DiabetesPedigree |	0.08 to 2.42  |	Model ignores this because the numbers seem tiny
BMI              |	18 to 67	  |Reasonable range

StandardScaler converts all features to the same scale (approximately -3 to +3) so the model compares them fairly. The critical rule is:

•fit_transform on training data — learns the average and range from training data
•transform on test data — applies the same rules WITHOUT looking at test data
•Never fit_transform on test data — that would be data leakage (cheating)

4. Model Audition — Testing Four Algorithms
Rather than picking one algorithm blindly, we test four and let the data decide which performs best on this specific problem.

Model               |	Accuracy |	Notes
Random Forest       |	77.92%	 |WINNER — most stable on small datasets
XGBoost             |	75.32%   |	Usually wins on larger data, needs more rows to shine
SVM	                |   75.32%	 |Tied with XGBoost, good at finding complex boundaries
Logistic Regression |	70.78%   |	Simplest model, weakest here but most explainable

Why Random Forest Won
The dataset only has 768 patients. XGBoost is a very powerful algorithm that needs large datasets to outperform simpler models. Random Forest builds 100 independent decision trees and lets them vote on small data this averaging effect provides stability that sequential boosting algorithms cannot match.



5. Experiments That Did Not Work
A key part of professional ML engineering is documenting what you tried and why it did not improve the model. These experiments are preserved in the code as commented-out sections with explanations.

Experiment 1 — Feature Engineering
Three new features were created from the existing data:
•BMI_Age = BMI multiplied by Age (older and heavier = higher risk)
•Glucose_Insulin = Glucose divided by Insulin (how well the body processes sugar)
•High_Risk = binary flag for patients with Glucose above 140 AND BMI above 30

Before Feature Engineering |	After Feature Engineering
Random Forest: 77.92%	   |    Random Forest: 72.08%

Result: Accuracy dropped by almost 6%. The engineered features added redundancy and noise rather than new information. The model already had access to BMI and Age separately — multiplying them together did not teach the model anything new, it just created confusion.
Lesson learned: More features does not always mean better results. Always measure the impact of changes.

Experiment 2 — GridSearchCV on XGBoost
GridSearchCV was used to systematically test combinations of XGBoost hyperparameters across 81 different configurations with 5-fold cross validation (405 total model fits).
•Best learning_rate: 0.01 (very small steps, careful learning)
•Best max_depth: 4 (slightly deeper trees than default)
•Best n_estimators: 300 (more trees than default 100)
•Best scale_pos_weight: 1 (no class imbalance correction needed)
•Best score achieved: 76.7%

Result: Even the perfectly tuned XGBoost could not beat the default Random Forest at 77.92%. This confirms that the dataset size is the limiting factor — not the choice of hyperparameters.

Experiment 3 — KNN Imputation
Instead of filling missing values with the column median, KNN Imputation looks at the 5 most similar patients and uses their values to fill the gap. This is more medically intelligent because a patient's missing insulin level can be estimated from patients who are similar in age, BMI, and glucose.
Result: Marginal improvement of approximately 0.5%. Not worth the added complexity for this project, but would be worth implementing on a larger dataset.

6. SHAP Explainability — The Key Differentiator
SHAP (SHapley Additive exPlanations) uses game theory mathematics to explain exactly why a model made a specific prediction. This transforms the model from a black box into a transparent medical tool.

Two Types of Explanation
Summary Plot                                  |	Waterfall Plot
Shows ALL 154 test patients	                  | Shows ONE specific patient
Answers: which features matter for everyone?  |	Answers: why was THIS patient flagged?
Red dots on right = high value increases risk | Red bars push toward diabetic diagnosis
Blue dots on left = high value decreases risk |	Blue bars push toward healthy diagnosis

Reading The Waterfall Plot — Patient #1
Feature          |	SHAP Value |	Medical Interpretation
Glucose	         |  +0.30      |	Blood sugar is significantly elevated — biggest risk factor
BMI	             |  -0.10      |	Body weight is below average for this patient — protective
Age              |	+0.07      |	Slightly older than average — minor risk increase
Pregnancies      |	+0.06      |	Number of pregnancies adds small risk
DiabetesPedigree |	-0.03      |	Family history is below average — small protection
BloodPressure    |	+0.00      |	No meaningful impact on this prediction

Starting from 34.7% average risk, Glucose adds 30% pushing toward diabetes while the below-average BMI pulls back 10%. After all factors are weighed the final prediction is 67% probability of diabetes.

Why This Matters In Medicine
A doctor can look at the waterfall plot and say: 'The model flagged this patient primarily because of elevated glucose. Their BMI is actually helping them. We should focus on blood sugar management.' This level of transparency is what separates medical ML tools from black boxes that nobody trusts.

7. Why 77% Is The Honest Ceiling
Several approaches were tried to push accuracy above 80%. None succeeded because the dataset itself has fundamental limitations:

•Only 768 patients — too small for complex algorithms to show their full power
•Only 8 features — missing critical medical information
•No HbA1c levels — the actual gold standard diabetes test is not in the dataset
•No diet or exercise data — major diabetes factors completely absent
•No medication history — current treatments affect all measurements
•No duration of symptoms — early vs late stage diabetes look different

Approach Tried         |	Result
Default Random Forest  |	77.92% — best result
Feature Engineering    |	72.08% — got WORSE
XGBoost GridSearchCV   |	76.7% — below baseline
KNN Imputation         |	~78.4% — marginal gain, not worth complexity

77% with SHAP explainability is more valuable in a medical context than 90% from a black box model that nobody can explain or trust.

8. How To Explain Different Patients
The waterfall plot can explain any patient in the test set. Change the patient_index parameter in the explain_patient function call:

# Patient 1 (index 0)
explain_patient(..., patient_index=0)

# Patient 5 (index 4)
explain_patient(..., patient_index=4)

# Any patient from 0 to 153 (154 test patients total)
explain_patient(..., patient_index=50)

9. Installation
pip install pandas numpy matplotlib scikit-learn xgboost shap

python diabetes_classifier.py


Diabetes Risk Predictor — ML Engineering Project Complete
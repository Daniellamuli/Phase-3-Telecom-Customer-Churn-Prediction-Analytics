<div align="center">
  <h1><strong>📱 SyriaTel Customer Churn Prediction</strong></h1>
  <p><em>Optimizing Revenue Retention through Advanced Predictive Analytics</em></p>
</div>

<hr />
<p align="center">
  <img src="Images/Syriatel-churn-prediction-strategy-Banner.jpg" 
       alt="SyriaTel Strategy Banner" 
       style="width: 48%; height: 250px; object-fit: cover; object-position: top; border-radius: 8px;">
  <img src="Images/SyriaTel Project Banner.jpg" 
       alt="Abstract Data Visualization" 
       style="width: 48%; height: 250px; object-fit: cover; border-radius: 8px;">
</p>

![churn](https://img.shields.io/badge/Predict-Churn-red)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Latest-orange)


## Overview
SyriaTel, a telecommunications company, faces a common but costly challenge: **customer churn** (the loss of clients to competitors). Losing customers not only reduces revenue but also increases acquisition costs to replace them; a significant financial drain in the telecommunications industry. This project utilizes machine learning to identify predictable patterns in customer behavior for SyriaTel. It aims to build a **binary classification model** that predicts whether a customer will stop doing business with SyriaTel in the near future. By identifying and predicting which customers are likely to leave "soon," we enable the business to take proactive retention measures such as targeted offers or improved service to retain them, ultimately protecting the bottom line. 

The analysis follows the full data science lifecycle: from business understanding and data exploration to model building, evaluation, and actionable recommendations. The final deliverable is a predictive model that helps SyriaTel reduce churn and maximize customer lifetime value.

---

## Business and Data Understanding

### The Stakeholder
The primary stakeholder is the SyriaTel Retention Department.

### Business Problem
It costs significantly more to acquire a new customer than to retain an existing one. Every customer who churns represents lost monthly revenue and a wasted acquisition cost.
SyriaTel’s leadership wants to understand **why customers leave** and **who is likely to leave next**. They need a tool that flags high-risk customers so that the retention team can intervene. The key business questions are:
- What behavioral patterns and early indicators signal that a customer is likely to churn?

- Which customer features (e.g., usage levels, service plans, customer service interactions) are the strongest predictors of churn?

- How accurately can we predict customer churn before it occurs?

- How can churn insights be translated into effective and cost-efficient customer retention strategies?

### Dataset
The dataset comes from SyriaTel’s customer records and contains **3333 observations** with **21 features**. Each row represents a unique customer, and the target variable is `churn` (True/False). The features include:

- **Demographics:** `state`, `area code`  
  → Provide geographic information about the customer’s location and regional classification.

- **Account Information:** `account length`, `international plan`, `voice mail plan`, `number vmail messages`  
  → Describe customer subscription details, tenure with the telecom provider, and access to additional service plans.

- **Usage Metrics:** Day, evening, night, and international minutes, number of calls, and corresponding charges  
  → Capture customer calling behavior and service usage patterns across different times of day.

- **Service Interaction:** `customer service calls`  
  → Represents the number of times a customer contacted support, which may indicate dissatisfaction or service-related issues.

- **Plan Details:** Presence of International or Voice Mail plans  
  → Highlight whether customers are subscribed to optional service features that may influence usage behavior and churn risk.

- **Unique Identifier:** `phone number` (dropped during modeling)  
  → Serves as a unique customer identifier but was removed since it does not contribute predictive value.

- **Target:** `churn (True/False)`  
  → Indicates whether the customer discontinued the telecom service.

Initial exploration revealed a **class imbalance**: only about 14.5% of customers churned. This imbalance is realistic—churn is a rare event—and influences our choice of evaluation metrics (prioritizing Recall over Accuracy). (Accuracy alone is misleading).

---

## Data Preparation
Before modeling, we performed several preprocessing steps to ensure data quality and prevent leakage:

1. **Feature Engineering**: Created `total_charge` and `avg_call_cost` to capture the overall financial weight of a customer.
2. **Removed irrelevant columns**: `phone number` (unique identifier) was dropped.
3. **Encoded categorical variables**:
   - Binary columns (`international plan`, `voice mail plan`) converted to 0/1.
   - `state` and `area code` were one‑hot encoded to preserve any geographic signal.
4. **Train/test split**: Data was split **before scaling** to avoid data leakage. We used a 80/20 split with stratification to maintain class proportions.
5. **Feature scaling**: Numeric features (minutes, calls, charges, etc.) were standardized using `StandardScaler` for models sensitive to feature scales (e.g., logistic regression) to perform optimally. Tree‑based models used unscaled data.

All transformations were fit **only on the training set** and then applied to the test set, ensuring realistic performance estimates.

---

## Exploratory Data Analysis (EDA)
We dove into the data to uncover patterns associated with churn. Here are some key findings:

## Exploratory Data Analysis (EDA)

We explored the dataset to understand how different customer characteristics and behaviors relate to churn. Key insights include:

- **Customer service calls:** Customers who churn tend to contact customer support much more frequently. On average, churners make **about twice as many service calls** as non-churners, suggesting that repeated issues or dissatisfaction may increase the likelihood of leaving.

- **International plan:** Customers subscribed to the International Plan churn at a significantly higher rate (**~42%**) compared to those without the plan (**~11%**). This may indicate higher expectations, pricing concerns, or service challenges among international users.

- **Total day usage:** Churners generally record **higher total day minutes and charges**, indicating that heavier daytime users may be more sensitive to pricing or service quality.

- **Voice mail plan:** Customers with a Voice Mail Plan appear to be more loyal, with a lower churn rate (**~9%**) compared to customers without the plan (**~16%**). This feature may increase perceived service value or engagement.

- **Account length:** There is **little difference in tenure** between customers who churn and those who stay. This suggests that how long a customer has been with the provider is not a strong indicator of churn on its own.
Correlation analysis confirmed that minutes and charges are perfectly correlated (as expected), so we could drop one set to reduce redundancy, but we kept them for interpretability.

---

## Modeling
We adopted an iterative modeling approach, starting with a simple interpretable baseline and progressing to more complex algorithms.

### Baseline Model: Logistic Regression
Logistic regression provides a clear view of feature importance and is highly interpretable. We used the scaled training data and applied L2 regularization (default). The model achieved:
- **Test Accuracy**: 86%
- **ROC‑AUC**: 0.82
- **Precision (churn)**: 0.62
- **Recall (churn)**: 0.44

While accuracy looks decent, recall is low—only 44% of actual churners were caught. This is because the model tends to predict the majority class.

### Second Model: Decision Tree
Decision trees capture non‑linear relationships and are still interpretable. We tuned `max_depth` and `min_samples_split` using grid search. The best tree:
- **Test Accuracy**: 91%
- **ROC‑AUC**: 0.85
- **Precision (churn)**: 0.73
- **Recall (churn)**: 0.61

Performance improved, but recall remained modest.

### Ensemble Models
We then experimented with ensemble methods to boost predictive power.

#### Random Forest
Random forest combines many trees and often yields robust results. After hyperparameter tuning (n_estimators, max_depth, min_samples_split), the final model:
- **Test Accuracy**: 93%
- **ROC‑AUC**: 0.94
- **Precision (churn)**: 0.88
- **Recall (churn)**: 0.72

Recall jumped to 72%, meaning we now correctly identify nearly three‑quarters of churners.

#### Gradient Boosting (XGBoost)
XGBoost is a powerful boosting algorithm. With tuning, it delivered similar performance:
- **Test Accuracy**: 93%
- **ROC‑AUC**: 0.95
- **Precision**: 0.86
- **Recall**: 0.74

### Model Selection
We selected **Random Forest** as the final model because:
- It achieves the best balance between precision and recall (F1‑score = 0.79).
- It provides built‑in feature importance, helping explain predictions.
- It is less prone to overfitting than a single tree.

---

## Evaluation
Given the business context, we prioritized **recall** over precision: missing a churner (false negative) costs more than a false positive (wasted retention offer). However, we also want reasonable precision to avoid annoying too many loyal customers.

Our final Random Forest model on the test set:
| Metric        | Value |
|---------------|-------|
| Accuracy      | 93%   |
| ROC‑AUC       | 0.94  |
| Precision     | 0.88  |
| Recall        | 0.72  |
| F1‑score      | 0.79  |

The confusion matrix shows:
- **True Negatives**: 569 (correctly predicted stayers)
- **False Positives**: 19 (stayers misclassified as churners)
- **False Negatives**: 39 (churners missed)
- **True Positives**: 100 (churners correctly identified)

With this model, SyriaTel could potentially retain **72% of customers who would otherwise churn** by targeting them with appropriate interventions.

### Feature Importance
The top predictors of churn are:
1. **Total day minutes** (and its corresponding charge)
2. **Customer service calls**
3. **International plan** subscription
4. **Total evening minutes**
5. **Number of voicemail messages** (negative association—more messages, less churn)

These insights guide retention strategies: focus on heavy daytime users, those who call customer service frequently, and international plan subscribers.

---

## Recommendations
Based on the model and EDA, we recommend SyriaTel take the following actions:

1. **Proactive outreach**: Use the model to score existing customers weekly. For those in the top risk decile, offer personalized incentives (e.g., discounts on daytime minutes, upgraded international plans).
2. **Improve customer service**: Since high call volume to customer service correlates with churn, analyze those interactions to identify root causes and improve resolution.
3. **Targeted promotions for international plan users**: These customers are at high risk—consider loyalty programs or plan adjustments.
4. **Monitor usage patterns**: Sudden spikes in daytime minutes could indicate dissatisfaction or a pending switch; trigger a check‑in.

---

## Limitations
- **Class imbalance**: Even though we used appropriate metrics, the model’s recall could be higher. Collecting more data on churners or using synthetic oversampling (SMOTE) might improve.
- **Static snapshot**: The data represents a single point in time. Churn behavior can change; the model should be retrained periodically.
- **Limited features**: We don’t have competitor pricing, customer sentiment, or network quality data, which could be even more predictive.
- **Geographic encoding**: One‑hot encoding states added many features; a more compact representation (e.g., grouping by region) could reduce dimensionality.

---

## Next Steps
- Deploy the model as a scoring API for integration with SyriaTel’s CRM.
- A/B test retention campaigns on high‑risk customers to measure impact.
- Collect additional data (e.g., contract length, payment method, competitor actions) to further refine predictions.
- Explore deep learning or advanced boosting algorithms if performance plateaus.

---

## Repository Structure
├── data/
│ └── bigml_59c28831336c6604c800002a.csv # raw data
├── notebooks/
│ └── syriatel_churn_analysis.ipynb # Jupyter Notebook with full analysis
├── images/ # plots and figures
├── README.md # this file
├── LICENSE
└── .gitignore


---

## Conclusion
This project demonstrates how machine learning can help a telecom company predict and mitigate customer churn. By identifying at‑risk customers early, SyriaTel can take targeted actions to improve retention, ultimately saving revenue and strengthening customer relationships. The final Random Forest model balances interpretability and performance, offering a practical tool for the business.

For a detailed walkthrough, check the [Jupyter Notebook](notebooks/syriatel_churn_analysis.ipynb).

---

**MIT License** – feel free to use and adapt this analysis.

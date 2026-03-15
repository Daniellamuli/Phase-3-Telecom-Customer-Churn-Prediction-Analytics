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

---
## Key Findings

### Exploratory Data Analysis Insights

Initial exploration revealed a **class imbalance**: only about 14.5% of customers churned. This imbalance is realistic—churn is a rare event—and influences our choice of evaluation metrics (prioritizing Recall over Accuracy). (Accuracy alone is misleading).
<p align="center">
  <img src="./Images/churn%20distribution.png" alt="Churn Distribution" width="600">
  <br>
  <i>Figure 1: Distribution of Customer Churn - Clear class imbalance is visible</i>
</p>

We explored the dataset to understand how different customer characteristics and behaviors relate to churn. Key insights include:

- **Customer service calls:** Customers who churn tend to contact customer support much more frequently. On average, churners make **about twice as many service calls** as non-churners, suggesting that repeated issues or dissatisfaction may increase the likelihood of leaving.

- **International plan:** Customers subscribed to the International Plan churn at a significantly higher rate (**~42%**) compared to those without the plan (**~11%**). This may indicate higher expectations, pricing concerns, or service challenges among international users.

- **Total day usage:** Churners generally record **higher total day minutes and charges**, indicating that heavier daytime users may be more sensitive to pricing or service quality.

- **Voice mail plan:** Customers with a Voice Mail Plan appear to be more loyal, with a lower churn rate (**~9%**) compared to customers without the plan (**~16%**). This feature may increase perceived service value or engagement.

- **Account length:** There is **little difference in tenure** between customers who churn and those who stay. This suggests that how long a customer has been with the provider is not a strong indicator of churn on its own.

### Multicollinearity Note
Correlation analysis confirmed that minutes and charges are perfectly correlated (as expected) so we could drop one set to reduce redundancy, but we kept them for interpretability.

<p align="center">
  <img src="./Images/correlation%20heatmap.png" alt="Correlation Heatmap" width="600">
  <br>
  <i>Figure 2: Correlation Matrix - Note perfect correlation between minutes and charges</i>
</p>

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

## Modeling  

To identify customers at risk of churning, we adopted an **iterative modeling strategy**, beginning with a simple interpretable baseline and progressively introducing more advanced models capable of capturing complex behavioral patterns.

Because the dataset is imbalanced (only ~14.5% churn), model performance was evaluated primarily using **Recall and F1-Score**, rather than Accuracy alone.

---

### Baseline Model — Logistic Regression  

Logistic Regression was used as a starting point due to its interpretability and ability to provide insight into how individual features influence churn probability.

**Performance on Test Set:**

- **Accuracy:** 76.6%  
- **ROC-AUC:** 0.819  
- **Precision (churn):** 0.353  
- **Recall (churn):** **0.732**  
- **F1-Score:** 0.477  

Although the model successfully identified many churners (high recall), it produced a large number of false positives. This reflects the limitations of linear decision boundaries when modeling complex customer behavior.

---

### Non-Linear Model — Decision Tree  

A Decision Tree was introduced to capture non-linear relationships and feature interactions. Hyperparameters such as `max_depth` and `min_samples_split` were tuned to improve generalization.

**Performance on Test Set:**

- **Accuracy:** 92.8%  
- **ROC-AUC:** 0.868  
- **Precision (churn):** 0.738  
- **Recall (churn):** 0.784  
- **F1-Score:** 0.760  

This model achieved a significantly better balance between identifying churners and reducing false alarms.

---

### Ensemble Model — Random Forest  

To further improve predictive stability and reduce overfitting, a Random Forest model was implemented. By aggregating multiple decision trees, the model captures complex patterns while improving generalization.

**Baseline Random Forest Performance:**

- **Accuracy:** 94.3%  
- **ROC-AUC:** 0.898  
- **Precision (churn):** **0.984**  
- **Recall (churn):** 0.619  
- **F1-Score:** 0.760  

The model delivered extremely high precision, meaning churn predictions were highly reliable. However, recall declined, indicating that some at-risk customers were still being missed.

---

### Random Forest Optimization  

Hyperparameter tuning using **GridSearchCV** was performed to improve recall while maintaining strong predictive performance.

**Best Parameters Identified:**

- `n_estimators = 100`  
- `max_depth = 20`  
- `min_samples_split = 2`  

**Tuned Random Forest Performance:**

- **Accuracy:** 94.6%  
- **ROC-AUC:** 0.901  
- **Precision (churn):** 0.969  
- **Recall (churn):** 0.650  
- **F1-Score:** 0.778  

Tuning improved model balance slightly, but recall remained lower than desired for churn intervention purposes.

---

### Final Model — Voting Classifier (Super Ensemble)  

To leverage the strengths of multiple models, a **Voting Classifier ensemble** was constructed. This model combines predictions from Logistic Regression, Decision Tree, and Random Forest to improve overall robustness.

**Performance on Test Set:**

- **Accuracy:** **95.4%**  
- **ROC-AUC:** **0.912**  
- **Precision (churn):** 0.893  
- **Recall (churn):** **0.773**  
- **F1-Score:** **0.829**  

The Voting Classifier achieved the strongest overall performance, providing the best trade-off between identifying churners and maintaining prediction reliability.

---

## Model Selection  

Although the Random Forest model produced the highest precision, the **Voting Classifier was selected as the final model** because it achieved the best balance between recall and overall predictive quality.

In the context of churn prediction, **missing a customer who is about to leave (false negative) is more costly than incorrectly flagging a loyal customer (false positive)**.  

With a recall of approximately **77%**, the final model enables SyriaTel to proactively identify the majority of customers at risk of churn and intervene with targeted retention strategies.

---

## Feature Importance  

Understanding which factors most strongly influence churn allows SyriaTel to move from prediction to **actionable retention strategy**.

The feature importance analysis from the final ensemble model highlights several key behavioral and service-related drivers of customer churn.

<div align="center">
  <img src="Images/feature%20importance.png" alt="Feature Importance" width="700"/>
  <p><em>Figure 4: Most influential features contributing to churn predictions.</em></p>
</div>

### Key Drivers of Customer Churn  

1. **Total Day Usage (Minutes / Charges)** ⭐  
   Daytime usage intensity emerged as the strongest predictor of churn. Customers with high daytime consumption are more likely to leave, potentially due to pricing sensitivity or unmet expectations regarding service value. These customers may represent business or high-dependency users who are more responsive to competitive offers.

2. **Customer Service Calls**  
   Frequent interactions with customer support serve as a strong signal of dissatisfaction. Multiple calls may indicate unresolved service issues, billing concerns, or declining trust in service quality.

3. **International Plan Subscription**  
   Customers enrolled in international plans show elevated churn risk. This suggests that the perceived value of international offerings may not align with customer needs or pricing expectations.

4. **Voice Mail Engagement**  
   Customers with higher voicemail usage or active voice mail plans tend to churn less frequently. This may reflect stronger service integration, higher engagement, or increased switching costs associated with bundled features.

5. **Evening Usage Patterns**  
   Evening call behavior also contributes to churn prediction, indicating that overall communication intensity across multiple time periods influences customer retention dynamics.

---

### Strategic Insight  

These findings suggest that churn risk is not driven by a single factor but rather by a **combination of usage intensity, service experience, and plan value perception**.  

By proactively monitoring these indicators, SyriaTel can design targeted retention initiatives aimed at high-risk customer segments before churn occurs.

---
## Recommendations  

Based on insights from exploratory analysis and predictive modeling, we propose a targeted retention strategy focused on **early intervention, service experience improvement, and value optimization for high-risk segments.**

---

### 1. 📞 Proactive Retention Outreach  

Deploy the churn prediction model in a production scoring pipeline to evaluate customers on a regular basis (e.g., weekly or monthly). Customers identified within the **highest risk decile** should receive proactive engagement such as:

- Personalized retention offers (discounts, plan upgrades, or loyalty rewards)
- Dedicated follow-up from retention specialists
- Usage-based recommendations to optimize plan suitability  

Early intervention increases the likelihood of preventing churn before customers fully disengage.

---

### 2. 🛠 Strengthen Customer Service Experience  

Frequent customer service interactions emerged as one of the strongest indicators of churn risk. To address this:

- Implement automated alerts when customers exceed **three support calls within a billing cycle**
- Route these customers to specialized “save teams” trained in retention
- Analyze support interactions to identify recurring pain points and improve first-contact resolution  

Improving service experience can directly reduce frustration-driven churn.

---

### 3. 💼 Optimize Offerings for High Daytime Usage Customers  

Customers with high daytime usage intensity demonstrate elevated churn risk, likely due to pricing sensitivity or unmet value expectations. Recommended actions include:

- Introducing targeted bundles such as **“Business Hours Unlimited”** plans  
- Triggering proactive plan upgrade recommendations when usage thresholds are exceeded  
- Offering loyalty incentives for consistently high-revenue customers  

These strategies help align perceived value with actual usage patterns.

---

### 4. 🌍 Reposition the International Plan  

Subscribers to international plans are significantly more likely to churn. SyriaTel should conduct a product value audit by:

- Revising international call pricing structures  
- Introducing bundled international features (e.g., roaming or data add-ons)  
- Offering region-specific discounts based on common calling destinations  
- Collecting targeted feedback to better understand customer expectations  

Enhancing the competitiveness of international offerings can improve retention within this strategically important segment.

---

### Strategic Takeaway  

By combining predictive analytics with targeted operational interventions, SyriaTel can transition from reactive churn management to a **proactive, data-driven retention strategy** that protects revenue and strengthens long-term customer relationships.

---

## Limitations

While the churn prediction model demonstrates strong performance, several limitations should be considered when interpreting results and planning future improvements.

### 1\. Class Imbalance

The dataset exhibits a significant imbalance between retained and churned customers. Although evaluation metrics such as **Recall and F1-Score** were prioritized and class weighting was applied during training, the model may still underperform in identifying all potential churners.

Future improvements could include:

*   Collecting additional data on churned customers
    
*   Applying advanced resampling techniques such as **SMOTE**
    
*   Exploring threshold tuning to further optimize recall
    

* * *

### 2\. Static Nature of the Dataset

The data represents a **snapshot of customer behavior at a single point in time**. In real-world telecommunications environments, churn dynamics evolve due to pricing changes, competitive actions, and seasonal usage patterns.

To maintain predictive accuracy, the model should be:

*   Retrained periodically using updated customer data
    
*   Monitored for performance drift
    
*   Integrated into a continuous learning pipeline
    

* * *

### 3\. Limited Feature Scope

Key drivers of churn may not be fully captured due to the absence of potentially informative variables such as:

*   Competitor pricing and promotional campaigns
    
*   Customer satisfaction or sentiment indicators
    
*   Network performance metrics (e.g., dropped calls, service outages)
    

Incorporating richer behavioral and experiential data could significantly enhance predictive power.

* * *

### 4\. High-Dimensional Geographic Encoding

The use of **one-hot encoding for state-level geography** increases feature dimensionality, which can introduce sparsity and reduce model efficiency.

Future iterations could improve representation by:

*   Grouping locations into regional clusters
    
*   Using target encoding or embedding-based techniques
    
*   Evaluating whether geographic granularity materially impacts churn prediction
    

* * *

### Key Reflection

Despite these limitations, the model provides a strong foundation for **data-driven retention strategies**. Addressing the above constraints represents a clear roadmap for improving both predictive performance and operational deployment readiness.

---

## Next Steps
- Deploy the model as a scoring API for integration with SyriaTel’s CRM.
- A/B test retention campaigns on high‑risk customers to measure impact.
- Collect additional data (e.g., contract length, payment method, competitor actions) to further refine predictions.
- Explore deep learning or advanced boosting algorithms if performance plateaus.

---

## Repository Structure

## 

    ├── data/
    │   └── bigml_59c28831336c6604c800002a.csv     # Raw churn dataset
    │
    ├── notebooks/
    │   └── syriatel_churn_analysis.ipynb           # Full end-to-end analysis notebook
    │
    ├── images/                                     # Saved plots used in README
    │   ├── churn_distribution.png
    │   ├── voting_classifier_confusion.png
    │   └── feature_importance.png
    │
    ├── presentation/
    │   └── Syriatel_Churn_Presentation.pdf         # Final stakeholder presentation
    │
    ├── requirements.txt                             # Python dependencies
    ├── README.md
    ├── LICENSE
    └── .gitignore
    

* * *

## How to Reproduce This Analysis

## 

To run this project locally, follow the steps below:

### 1️⃣ Clone the Repository

## 

    git clone https://github.com/your-username/syriatel-churn-prediction.git
    cd syriatel-churn-prediction
    

* * *

### 2️⃣ Create and Activate a Virtual Environment

## 

    python -m venv venv
    source venv/bin/activate        # macOS / Linux
    venv\Scripts\activate           # Windows
    

* * *

### 3️⃣ Install Dependencies

## 

    pip install -r requirements.txt
    

* * *

### 4️⃣ Launch Jupyter Notebook

## 

    jupyter notebook
    

Then navigate to:

    notebooks/syriatel_churn_analysis.ipynb
    

Run all cells sequentially to reproduce the full analysis, modeling workflow, and visualizations.

* * *

## Requirements

## 

Key libraries used in this project include:

*   pandas
    
*   numpy
    
*   matplotlib
    
*   seaborn
    
*   scikit-learn
    
*   imbalanced-learn
    

All dependencies and versions are listed in `requirements.txt`.

* * *

## Presentation

## 

A stakeholder-focused summary of findings and business recommendations is available in:

    presentation/Syriatel_Churn_Presentation.pdf
    

This deck highlights the key churn drivers, model performance, and strategic retention actions.

---

## Conclusion
This project demonstrates how machine learning can help a telecom company predict and mitigate customer churn. By identifying at‑risk customers early, SyriaTel can take targeted actions to improve retention, ultimately saving revenue and strengthening customer relationships. The final Random Forest model balances interpretability and performance, offering a practical tool for the business.

---

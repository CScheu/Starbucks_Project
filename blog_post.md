# Starbucks Offer Recommendation – Capstone Project

## High-Level Overview

This project investigates customer behavior in response to promotional offers provided by Starbucks. The core question is:

> **"Which types of offers are most effective for which types of customers?"**

The goal is to help Starbucks personalize their marketing campaigns to maximize customer engagement and offer completion rates. Using transaction, offer, and demographic data, this project explores patterns in user behavior, builds predictive models, and provides actionable insights for targeted marketing.

---

## Description of Input Data

The dataset for this project is a simulated Starbucks dataset provided as part of the Udacity Data Science Nanodegree Capstone. It consists of three main files:

1. **`portfolio.json`** – Contains metadata about each offer:
   - `offer_type` (BOGO, discount, informational)
   - `difficulty` (effort required to complete)
   - `duration` (validity period)
   - `reward` (monetary or points reward)

2. **`profile.json`** – Contains user demographic information:
   - `age`
   - `gender`
   - `income`
   - `member_since` (membership start date)

3. **`transcript.json`** – Contains event logs of user activity:
   - `offer received`
   - `offer viewed`
   - `offer completed`
   - `transaction` events

These datasets allow the project to link customer characteristics, offer types, and behavioral outcomes to understand and predict offer effectiveness.

---

## Strategy for Solving the Problem

The overall approach for this project can be summarized as follows:

1. **Data integration** – Merge demographic, offer, and transaction data to create a unified dataset.
2. **Feature engineering** – Define variables such as “offer viewed,” “offer completed,” and “offer influenced” (viewed before completion).
3. **Exploratory Data Analysis (EDA)** – Identify patterns in offer responsiveness across demographics.
4. **Modeling** – Build a classification model to predict the likelihood of offer influence.
5. **Evaluation** – Assess model performance using metrics relevant to binary classification.
6. **Insights and recommendations** – Translate findings into actionable strategies for personalized marketing.

---

## Discussion of the Expected Solution

The proposed solution workflow:

1. **Data Preprocessing**: Clean missing or inconsistent values, encode categorical variables, and engineer features.
2. **EDA**: Visualize completion and influence rates by age, gender, income, and offer type to identify trends.
3. **Model Building**: Train a Random Forest classifier using customer demographics and offer metadata to predict offer influence.
4. **Evaluation and Interpretation**: Analyze feature importance and model performance to identify key drivers of offer completion.
5. **Deployment Considerations**: Insights could inform targeted marketing campaigns in real-world applications.

---

## Metrics with Justification

The following metrics were used to evaluate model performance:

- **Accuracy** – Measures overall proportion of correctly classified instances.
- **Precision** – Measures correctness of positive predictions (important to avoid targeting uninterested customers).
- **Recall** – Measures ability to identify actual positive cases (important to maximize influenced customers).
- **F1 Score** – Balances precision and recall for a robust evaluation metric.

These metrics were chosen because the task is a binary classification problem (offer influenced vs. not influenced) and both false positives and false negatives have business relevance.

---

## Exploratory Data Analysis (EDA)

Key insights from the EDA include:

- **Offer Influence by Type**:
  - BOGO offers had higher influence among younger users and males.
  - Discount offers were more effective for older and higher-income users.
  - Informational offers had low overall influence but were slightly more effective for middle-income females.

- **Demographic Patterns**:
  - Income positively correlated with discount offer completion.
  - Younger customers had higher responsiveness to BOGO offers.
  
**Visualization Example**:

```python
import seaborn as sns
sns.barplot(data=rates, x='gender', y='offer_completed', hue='gender', palette='viridis', legend=False)
plt.ylabel('Offer Completion Rate')
plt.title('Offer Completion Rate by Gender')
plt.ylim(0, 1)
plt.show()
```

## Data Preprocessing

Steps included:

1. **Handling missing values**
   - Missing age values were replaced with the median.
   - Missing income values were replaced with the median.

2. **Filtering unrealistic entries**
   - Ages below 18 were removed.

3. **Feature engineering**
   - Created binary flags: `offer_viewed`, `offer_completed`, `offer_influenced`.
   - Calculated `days_since_member` from `member_since` column.

4. **Merging datasets**
   - Combined `profile`, `portfolio`, and `transcript` to form a single modeling dataset.

---

## Modeling

A **Random Forest Classifier** was chosen due to:

- Ability to handle categorical and numerical features without extensive preprocessing.
- Robustness to overfitting when tuned properly.
- Interpretability through feature importance.

**Pseudocode:**

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

## Hyperparameter Tuning

- **Technique used:** Grid Search
- **Parameters tuned:**
  - `n_estimators`: 50, 100, 200
  - `max_depth`: 5, 10, 20
  - `min_samples_split`: 2, 5, 10
- **Rationale:** Optimize model performance while preventing overfitting

---

## Results

**Random Forest achieved:**

- **Accuracy:** 0.72  
- **Precision:** 0.68  
- **Recall:** 0.65  
- **F1 Score:** 0.66  

**Key Features:** offer type, age, income, days since membership  

Insights support targeted marketing strategies based on demographic profiles.

---

## Comparison Table

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 0.68     | 0.65      | 0.62   | 0.63     |
| Random Forest        | 0.72     | 0.68      | 0.65   | 0.66     |
| Gradient Boosting    | 0.71     | 0.67      | 0.64   | 0.65     |

---

## Conclusion

- BOGO offers work best for younger, male customers.  
- Discount offers perform well for older, high-income customers.  
- Informational offers have limited impact but can influence certain segments.  
- Predictive modeling allows Starbucks to personalize marketing, improving offer engagement and reducing wasted promotions.

---

## Improvements

- Incorporate **time-series analysis** to capture seasonality or frequency effects.  
- Use **uplift modeling** to predict incremental effect of offers.  
- Enhance demographic data with **behavioral features** (past transaction frequency, total spend).  
- Test additional machine learning models (XGBoost, LightGBM) for potential performance gains.

---

## Acknowledgment

Special thanks to:

- **Udacity** for providing the dataset and capstone framework.  
- **Open-source libraries** (Pandas, Scikit-learn, Seaborn, Matplotlib) for data manipulation and visualization.  
- Mentors and peers who provided guidance during the project.

---

## How to Run the Project

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/starbucks-offer-capstone.git
cd starbucks-offer-capstone
```

2. **Install dependencies:**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

3. **Launch Jupyter Notebook:**

```bash
jupyter notebook Starbucks_Capstone.ipynb
```

4. **Follow the notebook to reproduce EDA, modeling, and results.**

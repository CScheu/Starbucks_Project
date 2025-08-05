# Starbucks Offer Recommendation – Capstone Project

## Project Overview

This project explores customer behavior in response to different types of promotional offers provided by Starbucks. Using a combination of transaction, offer, and demographic data, the project seeks to answer:

> **"Which types of offers are most effective for which types of customers?"**

This project was developed as part of the Udacity Data Science Nanodegree Capstone.

---

## Problem Statement

Starbucks wants to personalize the promotional offers it sends to customers. However, different users react differently to the same offer depending on age, gender, income, and offer type. The goal of this project is to:

- Identify which customers are most likely to be influenced by different types of offers.
- Determine demographic patterns in responsiveness to offers (BOGO, discount, informational).
- Build a predictive model to estimate whether a user will complete an offer.

---

## Data Sources

The dataset contains simulated data provided by Starbucks, split across three files:

- **portfolio.json** – metadata on each offer (type, difficulty, duration, etc.)
- **profile.json** – user demographic info (age, gender, income, membership date)
- **transcript.json** – events triggered by users (viewed, received, completed offers; transactions)

---

## Analysis & Methodology

### 1. Data Cleaning & Preprocessing
- Parsed and merged event logs to create a clear event history.
- Engineered flags to define when an offer was **viewed**, **completed**, and **influenced** (viewed before completing).
- Merged demographic and offer metadata with event logs.

### 2. Exploratory Data Analysis (EDA)
- Identified trends in offer completion and influence across:
  - Gender
  - Age groups
  - Income brackets
  - Offer types

### 3. Visualizations
- Bar plots and heatmaps of influenced rates across demographic segments.
- Distribution plots of income, age, and response rates.

### 4. Modeling
- Built a classifier to predict offer influence based on user and offer features.
- Evaluated model using accuracy, precision, recall, and feature importance.

---

## Results

- **BOGO offers** were more effective for younger users and males.
- **Discount offers** performed better with older and higher-income customers.
- **Informational offers** had low influence rates, but still impacted certain groups (e.g., females and middle-income users).
- The predictive model (Random Forest) achieved reasonable performance in classifying influenced offers, with key features being offer type, income, and age.

---

## Key Learnings

- Merging and aligning offer events over time was a non-trivial challenge and required detailed logic.
- Influence needs to be carefully defined to avoid misleading conclusions.
- Demographic segmentation provides clearer insights than aggregate statistics.

---

## 🔧 Technologies Used

- Python (Pandas, NumPy, Scikit-learn, Matplotlib)
- Jupyter Notebook
- JSON for data loading
- Git/GitHub for version control

---

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/starbucks-offer-capstone.git
   cd starbucks-offer-capstone

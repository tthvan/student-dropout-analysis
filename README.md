# Early Warning System for Student Dropout  
Weekly Time-Series Prediction Using Attendance, Engagement & Behavior Data

This project develops a machine learning–based early warning system that predicts which students are at risk of dropping out within the next several weeks. It uses weekly time-series features derived from zoom attendance logs, engagement trackings, lateness, and quiz scores.  
The goal is to generate **actionable, early alerts** so staff can detect on time.

---

## 1. Overview

Dropout events are extremely rare and usually occur after long periods of disengagement.  
The project focuses on predicting **early warning signals**, rather than the final dropout week.

Key challenges:
- Extreme **class imbalance** (e.g., <1% dropout weeks)
- **Time-series problems** (no using future data)
- Highly variable student histories (each student's time series has a different start and end points)
- Mixed data types (attendance, calls, quiz scores, timing)

The project uses ML models, including **tree-based gradient boosting (XGBoost)**, **Naive Bayes**, and **Random Forest** with thorough feature engineering and a **time-aware train/test split** to avoid leakage.

---

## 2. Workflow

### **Stage 1 — Data Preprocessing**
- Convert to **weekly rows** for each student, summarizing their behaviors by week.
- Remove all weeks **after** the student’s dropout week.
- Forward-fill academic scores only from past weeks.
- Standardize missing values and consistent week_start alignment.

---

### **Stage 2 — Feature Engineering**

Feature families used:
- **Attendance patterns:**  
  `present_flag`, `present_ratio_4w`, `absences_last4`, `current_absent_streak`
- **Lateness:**  
  `total_late_minutes_lag1`, `late_last4w`
- **Engagement:**  
  `attended_ratio_week_mean_4w`, `total_attended_minutes_w_mean_4w`
- **Communication:**  
  `called_count_w_sum_4w`, `called_failed_w_sum_4w`
- **Academic:**  
  `avg_score_mean_sem1`, `avg_score_mean_sem2`, and missing flags
- **Time features:**  
  `weeks_since_start`

The project will also use Recursive Feature Elimination (RFE) for feature selection and SMOTE for class rebalancing. 

---

### **Stage 3 — Deriving Target Variable**

Two deriving strategies are implemented:

1. **Event-week label** (`y_event`):  
   `1` only on the week of dropout.
    This variable is no longer used later on due to very few positive samples (ratio 99:1)

2. **Early-warning label** (`y_next5`):  
   `1` if the student will drop out in the next *5 weeks*.
   This increases positive examples and better reflects real life needs. Still, the class imbalance is stil strong (98:2)

---

### **Stage 4 — Train/Test Split**
Data is split in 80/20 regarding the number of students. In other words, 80% of students and their time-series will be used as Training set, and the remaining 20% is used as test set. 

---

### **Stage 5 — Modeling: XGBoost**

XGBoost is used with class imbalance weights.
Threshold was selected by optimizing **Fβ** (the weighted harmonic mean of precision and recall) with **β = 1.5** (favoring recall for early detection).

**Optimal threshold:** `0.8260`
**β value:** `1.50`

Other models will be carried out soon to compare with the current XGBoost model.

### Classification Metrics

| Metric      | Value  |
|-------------|--------|
| Accuracy    | **0.976** |
| Precision   | **0.441** |
| Recall      | **0.564** |
| F1-score    | **0.495** |
| PR-AUC      | **0.470** |
| ROC-AUC     | **0.927** |


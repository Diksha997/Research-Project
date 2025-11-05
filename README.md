# 💳 Detecting Fraudulent Transactions using Supervised Machine Learning

> A machine learning–based approach to identify fraudulent credit card transactions with high accuracy and interpretability.

---

## 📌 Overview

This project focuses on detecting fraudulent credit card transactions using **supervised machine learning algorithms**.
By analyzing transaction patterns and using techniques such as **SMOTE**, **Stratified K-Fold Cross-Validation**, and **GridSearchCV**, the goal is to build robust models that generalize well on unseen data.

---

## 🧠 Objectives

* Build multiple classification models to detect fraudulent transactions.
* Handle data imbalance using **Synthetic Minority Oversampling Technique (SMOTE)**.
* Optimize models using **GridSearchCV** for best hyperparameters.
* Evaluate models using key metrics — Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
* Compare model performances and identify the most effective algorithm.

---

## 🧩 Dataset

* **Source:** [Kaggle – Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Details:** 284,807 transactions made by European cardholders, of which 492 are fraudulent.
* Features are numerical and anonymized (via PCA transformation) to ensure privacy.
* Highly imbalanced dataset — approximately 0.172% of transactions are fraudulent.

---

## ⚙️ Tools & Technologies

* **Language:** Python
* **Environment:** Google Colab
* **Libraries:**

  * `pandas`, `numpy` – Data manipulation
  * `matplotlib`, `seaborn` – Data visualization
  * `scikit-learn` – Model building, evaluation, and optimization
  * `imbalanced-learn` – Handling class imbalance (SMOTE)
  * `xgboost` – Advanced ensemble model

---

## 🧪 Methodology

1. **Data Preprocessing**

   * Handle missing values (if any).
   * Normalize the data for consistent feature scaling.
   * Apply **SMOTE** to balance minority (fraud) class.

2. **Model Development**

   * Implemented algorithms: **Logistic Regression**, **Decision Tree**, **Random Forest**.
   * Used **Stratified K-Fold Cross-Validation** to avoid overfitting.
   * Optimized hyperparameters using **GridSearchCV**.

3. **Evaluation Metrics**

   * Accuracy, Precision, Recall, F1-Score, ROC-AUC
   * Confusion Matrix & ROC Curve visualization

---

## 📊 Results

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time (s) |
| :------------------ | :------- | :-------- | :----- | :------- | :------ | :---------------- |
| Logistic Regression | 0.98     | 0.99      | 0.97   | 0.98     | 0.9975  | 115.7             |
| Decision Tree       | 1.00     | 1.00      | 1.00   | 1.00     | 0.9985  | 362.8             |
| Random Forest       | 1.00     | 1.00      | 1.00   | 1.00     | 0.9999  | 1763.8            |

**Insights:**

* Logistic Regression achieved **98% accuracy**, showing good linear separability.
* Tree-based models (Decision Tree & Random Forest) achieved near-perfect results.
* **Random Forest** had the **highest ROC-AUC (0.9999)** but required the **most training time**.

---

## 🏁 Outcome

* Successfully built and compared multiple supervised ML models.
* Identified **Random Forest** as the best performing model for fraud detection.
* Provided insights into model behavior and performance trade-offs.
* The trained model is saved as `best_model_random_forest.joblib` for deployment or further testing.


---

## 🚀 How to Run

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/Fraud-Detection-Using-ML.git
   cd Fraud-Detection-Using-ML
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook:

   ```bash
   jupyter notebook Fraud_Detection_Final.ipynb
   ```

---

## 📜 License

This project is open-sourced under the **MIT License**. You are free to use, modify, and distribute with proper credit.

---

## 💬 Acknowledgements

* **Dataset:** ULB Machine Learning Group, Belgium (via Kaggle).
* Developed as part of an academic research project under **IGDTUW**.

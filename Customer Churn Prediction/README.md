
# Customer Churn Prediction Using XGBoost Classifier

This project predicts customer churn — whether a customer will discontinue a service — using an **XGBoost Classifier**, a powerful gradient boosting algorithm. The model is trained on customer behavior and subscription data to help businesses identify at-risk customers and improve retention strategies.

---

##  Project Overview

Customer churn significantly impacts business growth and revenue.
The objective of this project is to:

* Analyze customer demographics and usage patterns
* Build a predictive model using **XGBoost**
* Evaluate model performance and feature importance

This notebook covers every steps 
**Exploratory Data Analysis** 
**Data Visualization**
**Data Preparation**
**Model Training**
**Model Evaluation** 


---

##  Technologies Used

* **Python**
* **Pandas**, **NumPy** – data cleaning and manipulation
* **Matplotlib**, **Seaborn** – data visualization
* **Scikit-learn** – preprocessing and metrics
* **XGBoost** – core machine learning algorithm
* **Google Colab**

---

##  Dataset

The dataset used for this project contains customer-level information such as:

* **CustomerID**
* **Gender**
* **Age**
* **Tenure** – number of months with the company
* **HasCrCard** - has creditcard or not
* **Estimated Salary**
* **Churn** – target variable (`Yes` or `No`)


##  Model Development

### 1. Data Preparation

* Missing value handling
* Label encoding for categorical variables
* Feature scaling and selection
* Train-test split (typically 80:20)

### 2. Model Training (XGBoost)

* Algorithm: **Extreme Gradient Boosting (XGBClassifier)**
* Key parameters tuned:

  * `n_estimators`
  * `max_depth`

* Used **GridSearchCV** for hyperparameter optimization.

### 3. Model Evaluation

* Metrics: 
**Accuracy**
**Precision**
**F1-score**

---

##  Results

* The XGBoost model achieved an test accuracy of **~86%** 

##  License

This project is licensed under the MIT License.

---

###  Author

Dhenuka Dudde (https://github.com/Dhenuka45)

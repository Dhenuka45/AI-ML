# Titanic Survival Prediction – Decision Tree Classifier

This project explores the tragic **Titanic disaster** through data and machine learning. A **Decision Tree Classifier** is built to predict whether a passenger survived based on their personal and travel details.

---

##  Objective

The aim of this project is to apply data preprocessing, feature engineering, and a supervised classification model to predict passenger survival outcomes.

---

##  Key Steps

1. **Data Analysis:** Explore passenger demographics, survival rates, and relationships between features like gender, class.
2. **Data Cleaning/Preparation:** Handle missing values, encode categorical variables, and normalize numerical features.
3. **Model Training:** Train a Decision Tree Classifier using `scikit-learn`.
4. **Model Evaluation:** Measure model performance with accuracy, confusion matrix, and feature importance visualization.

---

##  Model Details

* **Algorithm:** Decision Tree Classifier
* **Libraries Used:**

  * pandas, numpy
  * matplotlib, seaborn
  * scikit-learn

The model identifies key predictors such as passenger **sex**, **class (Pclass)**, and **age** as the most influential features for survival.

---

##  Results

The trained Decision Tree achieved a test accuracy of approximately **84%**.


---


## Dataset

It includes passenger data such as:

* Passanger Id, Name, Sex, Age
* Ticket, Fare, Cabin, Embarked
* Pclass, SibSp, Parch
* Survived (target variable)

---

##  Future Improvements

* Experiment with other algorithms (Random Forest, XGBoost)
* Apply feature scaling and cross-validation
* Use GridSearchCV for hyperparameter tuning

---

##  License

This project is released under the [MIT License](LICENSE).

---

### Author

Dhenuka Dudde (https://github.com/Dhenuka45)

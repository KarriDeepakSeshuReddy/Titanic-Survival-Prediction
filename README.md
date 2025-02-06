# Titanic-Survival-Prediction
This project applies Logistic Regression to predict the survival of passengers on the Titanic. The dataset is preprocessed to handle missing values, encode categorical features, and train a machine learning model.

Dataset
The dataset used is the Titanic dataset (titanic.csv), which contains passenger details like age, gender, fare, class, etc. The goal is to predict the Survived column (0 = No, 1 = Yes).

Preprocessing Steps
Handling Missing Values:
Age: Filled missing values with the median age (28).
Embarked: Filled missing values with the most frequent category ('S').
Dropped Cabin and Ticket due to excessive missing values.

Feature Engineering:
Encoded categorical variables (Sex, Embarked) using LabelEncoder.
Created a new family feature by summing SibSp and Parch.
Dropped unnecessary columns (Name, PassengerId, SibSp, Parch, Embarked).

Model Training
Used Logistic Regression to classify survivors.
Split dataset into 70% training and 30% testing.
Used GridSearchCV to tune hyperparameters.

Results
Accuracy Score: Computed on test data.
Best Parameters: Found using GridSearchCV.

Dependencies
Ensure you have the following Python libraries installed:
pip install pandas numpy seaborn matplotlib scikit-learn
Running the Code
Run the script using Python:
python titanic.py
Visualization
Countplot for Embarked vs. Survived.
Correlation Heatmap of numerical features.
Future Improvements
Try other ML models like Decision Trees or Random Forest.
Feature scaling for better model performance.
Engineering more features (e.g., Title extraction from Name).

Author
Deepak 🚀



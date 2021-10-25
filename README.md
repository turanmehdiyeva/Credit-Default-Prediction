# Credit-Default-Prediction: Project overwiev
- Initialize Libraries, Load Data & Preprocess¶
- Exploratory Data Analysis and Visualization
- Predictive Modeling
# Resources Used
- https://www.kaggle.com/pratjain/credit-card-default
# Data Cleaning
- made it readible in Python which was previously prepared in R language
- filled missing variables
- made variables and columns follow the same naming convention
# EDA
- I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the pivot tables.
![image](https://user-images.githubusercontent.com/85342455/138645363-705c3daf-bbf2-43ab-97ea-29c9d889a048.png)
![image](https://user-images.githubusercontent.com/85342455/138645437-9adc286f-a112-4463-8bc8-f961bd67cad5.png)
![image](https://user-images.githubusercontent.com/85342455/138645491-6bbb562e-f315-4da5-aced-0b8dfc95b30b.png)
# Model Building
First, I transformed the categorical variables into dummy variables with the help of DictVectorizer. I also split the data into train, tests and validation sets with a test size of 20%.

I tried three different models and evaluated them with roc_auc_score. I chose roc_auc_score because it is relatively easy to understand and correct observation about the model.

I tried three different models:

- Logistic Regression: roc_auc_score = 0.60 (overfitting)
- Decision Tree Classifier: roc_auc_score = 0.75
- Random Forest: roc_auc_score = 0.78

#imports used for model file
import sklearn.preprocessing
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

def model_columns(train,validate,test):
  '''This function will provide my models with the correct features to run for their x and y values'''

  # features that will be used for x columns in modeling
  features = ['tech_support_Yes','contract_type_One year', 'contract_type_Two year','internet_service_type_Fiber optic']
    
  # setting the x and y values for my train, validate and test sets 
  X_train = train[features]
  y_train = train['has_churned']

  X_validate = validate[features]
  y_validate = validate['has_churned']

  X_test = test[features]
  y_test = test['has_churned']

    
  return X_train, X_validate, y_train, y_validate, X_test, y_test

def baseline(y_train,y_validate):
  ''' this function will generate the baseline model and print its performance '''

  # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
  y_train = pd.DataFrame(y_train)
  y_validate = pd.DataFrame(y_validate)

  # 1. Predict property_value_pred_mean
  prop_value_pred_mean = y_train['property_value'].mean()
  y_train['prop_value_pred_mean'] = prop_value_pred_mean
  y_validate['prop_value_pred_mean'] = prop_value_pred_mean

  # 2. compute prop_value_pred_median
  prop_value_pred_median = y_train['property_value'].median()
  y_train['prop_value_pred_median'] = prop_value_pred_median
  y_validate['prop_value_pred_median'] = prop_value_pred_median

  # 3. RMSE of prop_value_pred_median
  rmse_baseline_train = mean_squared_error(y_train.property_value, y_train.prop_value_pred_median)**(1/2)
  rmse_baseline_validate = mean_squared_error(y_validate.property_value, y_validate.prop_value_pred_median)**(1/2)

  print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

def decision_tree_model(X_train, X_validate, y_train, y_validate):
    '''this function will create my decision tree model and print its performance'''

    # Make the model
    tree1 = DecisionTreeClassifier(max_depth=3, random_state=100)

    # Fit the model train data
    tree_train = tree1.fit(X_train, y_train)

    # Fit the model on validate data
    tree_validate = tree1.fit(X_validate, y_validate)

    # print out of model results 
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(tree1.score(X_train, y_train)))

    print('Accuracy of Decision Tree classifier on validate set: {:.2f}'
      .format(tree1.score(X_validate, y_validate)))

def random_forrest_model(X_train, X_validate, y_train, y_validate):
    '''this function will create my random forrest model and print its performance'''

    # creating Random Forrest classifier model
    rf = RandomForestClassifier(
    min_samples_leaf=1, 
    max_depth=10, 
    random_state=100)

    # fit model to train data
    rf.fit(X_train, y_train)

    # evaluating model
    rf.score(X_train, y_train)

    # printing model results
    print('Accuracy of Random Forrest classifier on training set: {:.2f}'
      .format(rf.score(X_train, y_train)))
    print('Accuracy of Random Forrest classifier on validate set: {:.2f}'
      .format(rf.score(X_validate, y_validate)))
    
def knn_model(X_train, y_train):
    '''this function will create my Knn model and print its evaluation'''

    # creating knn model
    knn = KNeighborsClassifier(n_neighbors=5)

    #fitting model on train data
    knn.fit(X_train, y_train)

    #printing test results 
    print('Accuracy of KNN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))

def logistic_regression_model(X_train, X_validate, y_train, y_validate):
    '''this function will create my logistic regression model and print its evaluation'''

    # Define the logistic regression model
    logit = LogisticRegression(random_state=100)

    # fit model on train data
    logit.fit(X_train, y_train)

    # making predictions on train 
    y_pred = logit.predict(X_train)

    #printing model results
    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
     .format(logit.score(X_train, y_train)))
    print('Accuracy of Logistic Regression classifier on validate set: {:.2f}'
     .format(logit.score(X_validate, y_validate)))

def test_model(X_test, y_test):
    '''this function will run my best model and print its performance'''
    
    # define the logistic regression model
    logit = LogisticRegression(random_state=100)

     # fit model on features
    logit.fit(X_test, y_test)

    # printing model results 
    print('Accuracy of Logistic Regression classifier on test set: {:.2f}'
     .format(logit.score(X_test, y_test)))
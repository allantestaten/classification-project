import pandas as pd
import numpy as np
from env import get_db_url
import matplotlib.pyplot as plt
import seaborn as sns
from pydataset import data
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#------------------- ACQUIRE TELCO DATA -------------------#

def new_telco_data():
    '''
    This function retrieves the telco data from the Codeup db.
    '''
    sql_query = """
                select * from customers
                join contract_types using (contract_type_id)
                join internet_service_types using (internet_service_type_id)
                join payment_types using (payment_type_id)
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))

    return df


#------------------- PREPARE TELCO DATA -------------------#

def split_telco_data(df):
    '''
    This function will split my data into my groups of train, validate and test as dataframes.
    I will use churn as my stratify because it is my target.
    '''
    train_validate, test = train_test_split(df, test_size=.1, 
                                        random_state=123, 
                                        stratify=df.churn)
    train, validate = train_test_split(train_validate, test_size=.166666, 
                                   random_state=123, 
                                   stratify=train_validate.churn)
    return train, validate, test

def prep_telco_data(df):
    '''
    This function will prepare my data for my machine learning model 
    '''
    # Removing null values stored as whitespace    
    df['total_charges'] = df['total_charges'].str.strip()
    df = df[df.total_charges != '']
    
    # Changing total charges from object to float
    df['total_charges'] = df.total_charges.astype(float)
    
    # Encoding binary categorical variables to 1 or 0
    df['is_female'] = df.gender.map({'Female': 1, 'Male': 0})
    df['has_partner'] = df.partner.map({'Yes': 1, 'No': 0})
    df['has_dependents'] = df.dependents.map({'Yes': 1, 'No': 0})
    df['has_phone_service'] = df.phone_service.map({'Yes': 1, 'No': 0})
    df['has_paperless_billing'] = df.paperless_billing.map({'Yes': 1, 'No': 0})
    df['has_churned'] = df.churn.map({'Yes': 1, 'No': 0})
    
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type']], dummy_na=False, \
                              drop_first=True)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    
    # split the data
    train, validate, test = split_telco_data(df)
    
    return train, validate, test
       
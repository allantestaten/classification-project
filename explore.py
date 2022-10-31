#imports used for explore file
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

def get_bar_churn(train):
    '''this function will produce a bar chart of churn'''

    #identifying data frame to be used in chart
    df = train

    #generates count plot 
    sns.countplot(x=df["churn"]).set(title='Churned Customers')

    #creates percentage of churned and not churned customers 

def churn_percentages(train):
    '''this fucntion will produce the percentage of churned and not churned customers'''
    
    print(train.churn.value_counts(normalize=True))

def get_churn_and_phone(train):
    '''this function will produce the bar chart of phone service and churn'''

    #generates count plot
    sns.countplot(data=train, x="phone_service", hue="churn").set(title='Churn And Phone Service')

def get_phone_chi(train):
    '''this function will run and print the results of the chi square test'''
   
    #creating cross tab to be used for statistical test 
    observed = pd.crosstab(train.churn, train.phone_service)

    #generating statistical test and results 
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    #printing results of chi square test
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p}')

def get_tech_support_graph(train):
    '''this function will produce the bar chart for tech support and churn'''

    #generates count plot
    sns.countplot(data=train, x="tech_support", hue="churn").set(title='Technical Support Helps')

def get_tech_support_chi(train):
    '''this function will run and print the results of the chi square test'''
   
    #creating cross tab to be used for statistical test 
    observed = pd.crosstab(train.churn, train.tech_support)

    #generating statistical test and results 
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    #printing results of chi square test
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p}')

def get_internet_graph(train):
    '''this function will produce the bar chart for internet type and churn'''
    
    #creates count plot
    sns.countplot(data=train, x="internet_service_type", hue="churn").set(title='Churn and Internet Service Type ')

def get_internet_type_chi(train):
    '''this function will run and print the results of the chi square test'''
   
    #creating cross tab to be used for statistical test 
    observed = pd.crosstab(train['churn'], train['internet_service_type'])

    #generating statistical test and results 
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    #printing results of chi square test
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p}')

def get_fiber_chi(train):
    '''this function will run and print the results of the chi square test'''
   
    #creating cross tab to be used for statistical test 
    observed = pd.crosstab(train['churn'], train['internet_service_type_Fiber optic'])

    #generating statistical test and results 
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    #printing results of chi square test
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p}')

def get_contract_graph(train):
    '''this function will produce the bar chart for contract type and churn'''
    
    #creates count plot
    sns.countplot(data=train, x="contract_type", hue="churn").set(title='Contract Type Matters')

def get_contract_chi(train):
    '''this function will run and print the results of the chi square test'''
   
    #creating cross tab to be used for statistical test 
    observed = pd.crosstab(train['churn'], train['internet_service_type'])

    #generating statistical test and results 
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    #printing results of chi square test
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p}')
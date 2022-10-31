#imports used to acquire the data
import pandas as pd
import numpy as np
import os
from env import get_db_url


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
    
    # Read in data from Codeup database and convert it into a dataFrame
    df = pd.read_sql(sql_query, get_db_url('telco_churn'))
    
    return df

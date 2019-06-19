import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
%matplotlib inline


#remove outliers
def remove_outliers(df):  
    pd_copy = pd.DataFrame(df, copy=True)
    for col in df.columns:
        try:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3-q1 #Interquartile range
            fence_low  = q1-1.5*iqr
            fence_high = q3+1.5*iqr
            pd_copy.loc[(pd_copy[col] < fence_low) | (pd_copy[col] > fence_high),col] = np.nan
        
        except:
            pd_copy[col] == df[col]
    
    return pd_copy

df = remove_outliers(df)  

#clean data:
def clean_data(df):
    #check duplicates... BUT duplicates are a result of the same house being sold a second or third time 
    #yr_renovated column has both nan and 0.0 filler values... change all to nan so 0.0 doesn't skew data
    df['yr_renovated'] = df['yr_renovated'].replace(0.0, np.nan)

    df['waterfront'] = df['waterfront'].fillna(0.0)

    #sqft_basement in string format, convert and replace 0.0 with null to find median without skewed outliers
    df['sqft_basement'] = pd.to_numeric(df['sqft_basement'], errors='coerce')
    df['sqft_basement'] = df['sqft_basement'].replace(0.0, np.nan)
    df['sqft_basement'] = df['sqft_basement'].fillna(df['sqft_basement'].median())

    return df
    

def log_transform(df, cols):
    ''' takes in dataframe and cols to log transfrom and returns 
        the dataset with the log transformed columns dropped the regular
        col'''
    df_copy = df.copy()
    for col in cols:
        df_copy[col] = np.log(df_copy[col])
        df_copy['log_{}'.format(col)] = df_copy[col]
        df_copy.drop(col, axis=1, inplace=True)
    return df_copy
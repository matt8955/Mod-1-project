import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy as sp

#remove outliers
def remove_outliers(df, cols):  
    ''' takes in df and cols we want to remove outliers from
    and returns a cleaned df'''
    pd_copy = pd.DataFrame(df, copy=True)
    for col in cols:
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

#clean data:
def clean_data(df):
    '''cleans yr_renovated, sqft_basement, watefront because 
       they had placeholders that skewed the data'''

    #yr_renovated column has both nan and 0.0 filler values... change all to nan so 0.0 doesn't skew data
    df['yr_renovated'] = df['yr_renovated'].replace(0.0, np.nan)

    df['waterfront'] = df['waterfront'].fillna(0.0)

    #sqft_basement in string format, convert and replace 0.0 with null to find median without skewed outliers
    df['sqft_basement'] = pd.to_numeric(df['sqft_basement'], errors='coerce')
    df['sqft_basement'] = df['sqft_basement'].replace(0.0, np.nan)
    df['sqft_basement'] = df['sqft_basement'].fillna(df['sqft_basement'].median())
    
    return df
    
import numpy as np #this is imported above. Should we delete?

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
                 
    
def scatter_one_vs_all(df, column):
    ''' a data frame and one column and plots scatters for all cols and
    against one var only works on current data sets # of features'''
    fig, ax = plt.subplots(5,4, figsize=(30,30))
    l = list(df.columns)
    l.remove(column)
    i = 0 #to track col index
    for m in range(5):
        for n in range(4):
            sns.scatterplot(x=l[i],y=column,data=df,ax=ax[m][n])
            i += 1
    return 0

def jarque_bera(depend, df):
    '''Tests for normality for features we are interested in using '''
    features = list(df.columns)
    features.remove(depend) #make col of all features to loop across
    fig, ax = plt.subplots(5,4, figsize=(30,30))
    i=0
    for m in range(5):
        for n in range(4):
            f = '{}~{}'.format(depend, features[i])
            model = smf.ols(formula=f, data=df).fit()
            resid1 = model.resid
            sm.qqplot(resid1, dist=sp.stats.norm, line='45', fit=True, ax=ax[m][n])
            ax[m][n].set_title('{}'.format(features[i]))
            i += 1
    return 0

# # JB test for TV
#     name = ['Jarque-Bera','Prob','Skew', 'Kurtosis']
#     test = sms.jarque_bera(model.resid)
#     return list(zip(name, test))                                               
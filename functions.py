import numpy as np
#clear outliers
def clean_outliers(df):
    for col in df.columns:
        q1 = df['{}'.format(col)].quantile(0.25)
        q3 = df['{}'.format(col)].quantile(0.75)
        iqr = q3-q1 #Interquartile range
        fence_low  = q1-1.5*iqr
        fence_high = q3+1.5*iqr
        df_out = df.loc[(df['{}'.format(col)] > fence_low) & (df['{}'.format(col)] < fence_high)]
        return df_out
    
#take in list of columns in def and log transform
def log_transform(df, cols):
    ''' takes in dataframe and cols to log transfrom and returns 
        the dataset with the log transformed columns dropped the regular
        col'''
    for col in cols:
        df[col] = np.log(df[col])
        df['log_{}'.format(col)] = df[col]
        df.drop(col, axis=1, inplace=True)
    return df
        
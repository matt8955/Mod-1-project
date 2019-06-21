import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy as sp
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import math



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
            df = pd_copy

        except:
            pd_copy[col] == df[col]

    return df

#clean data:
def clean_data(df):
    '''cleans yr_renovated, sqft_basement, watefront because
       they had placeholders that skewed the data'''

    #yr_renovated column has both nan and 0.0 filler values... change all to nan so 0.0 doesn't skew data
    df['yr_renovated'] = df['yr_renovated'].replace(0.0, np.nan)
    df['waterfront'] = df['waterfront'].fillna(0.0)
    df['view'] = df['view'].fillna(0.0)
    df['sqft_basement'] = df['sqft_basement'].replace('?', np.nan)

    return df



def replace_null_w_median(list_columns, df):
    df_copy = df.copy()
    for col in list_columns:
        if col in df.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            df_copy[col] = df_copy[col].replace(0.0, np.nan)
            df_copy[col] = df_copy[col].fillna(df[col].median())

        else:
            return print('inputted columns not in dataframe-- try again')

        return df_copy


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
    df_copy = df.copy()
    fig, ax = plt.subplots(5,4, figsize=(30,30))
    
    l = list(df_copy.columns)
    l.remove(column)
    i = 0 #to track col index
    for m in range(5):
        for n in range(4):
            sns.scatterplot(x=l[i],y=column,data=df_copy,ax=ax[m][n])
            i += 1
    return 0


def single_regression_plot(df, column):
    '''takes in df and tries to graph every independent var against the
    column variable if seaborn cant plot it, revert to scatter'''
    df_copy = df.copy()
    l = list(df_copy.columns)
    l.remove(column)
#     #automate amount of subplot rows
#     c = len(l)
    fig, ax = plt.subplots(5,4, figsize=(30,30))
    i = 0 #to track col index
    for m in range(5):
        for n in range(4):
            try:
                sns.regplot(x=l[i],y=column,data=df_copy,ax=ax[m][n])
                i += 1
            except:
                sns.scatterplot(x=l[i],y=column,data=df_copy,ax=ax[m][n])
                i += 1
    return


def qq_plot(depend, df):
    df_copy = df.copy()
    features = list(df.columns)
    features.remove(depend) #make col of all features to loop across
    fig, ax = plt.subplots(5,4, figsize=(30,30))
    i=0
    for m in range(5):
        for n in range(4):
            f = '{}~{}'.format(depend, features[i])
            model = smf.ols(formula=f, data=df_copy).fit()
            resid1 = model.resid
            sm.qqplot(resid1, dist=sp.stats.norm, line='45', fit=True, ax=ax[m][n])
            ax[m][n].set_title('{}'.format(features[i]))
            i += 1
    return



def histogram(df):
    return df.hist(bins=50, figsize=(20,15))




# # JB test for TV
#     name = ['Jarque-Bera','Prob','Skew', 'Kurtosis']
#     test = sms.jarque_bera(model.resid)
#     return list(zip(name, test))

def test_predictors(list_of_features, df, num_pred):
    '''input list of features, df, numbers of predictors you want to compare against log_price'''
    predictors = df.reindex(columns=list_of_features)

    linreg = LinearRegression() #create linear regression object
    selector = RFE(linreg, n_features_to_select = num_pred) 
    selector = selector.fit(predictors, df['log_price'])

    selector_list = selector.ranking_
    answer_list = []
    estimators = selector.estimator_


    for i in range(0,len(selector_list)):
         answer_list.append(f'{list_of_features[i]} - {selector_list[i]}')

    return (answer_list, estimators.coef_, estimators.intercept_)


from statsmodels.formula.api import ols

def create_model():
    f = 'log_price~ log_sqft_living + waterfront + grade + condition + log_sqft_living15 + view + yr_built + zipcode'
    model = ols(formula = f, data = df).fit()
    return model.summary()

def get_statistics(features, depend, df):
    '''put in list of cols in df or all cols and 
    depend var will return test stas for each var'''
    df_copy = df.copy()
    features_dict = {} #list to store stats by feature 
    features = list(features)
    if depend in features:
        features.remove(depend) #remove depend var
    for feature in features:
        f = '{}~{}'.format(depend, feature)
        model = ols(formula=f, data=df_copy).fit()
        feature_dict = {'r_squared': model.rsquared, 'pvalue' : model.pvalues[1]}
        features_dict.update({feature :feature_dict})
    return features_dict

def actual_vs_predicted_df(df):
    '''makes data frame of actual vs. predicted values and returns the data frame as df_predicted'''
    
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train)

    coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
    coeff_df

    y_pred = regressor.predict(X_test)

    df_predicted = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    return df_predicted



def bar_error():
    '''makes a bar chart of first 25 entries in data set for actual price vs predicted price... '''
    df1 = df_predicted.head(25)
    
    df1.plot(kind='bar',figsize=(10,8), color=['lightSkyBlue', 'sandyBrown'])
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='lightgrey')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title('Actual vs. Predicted Price for First 25 Houses in Data Set', fontdict=None, loc='center', pad=None)
    plt.show()
    return


def regression_plot():
    '''creates a regression plot of actual vs. predicted values.'''
    import seaborn as sns; sns.set(style="white", color_codes=True)
    g = sns.jointplot(df_predicted.Actual, df_predicted.Predicted, 
                  data=df_predicted, kind='reg', 
                  joint_kws={'line_kws':{'color':'rosyBrown'}})
    g.fig.suptitle("Predicted vs. Actual Values Regression Plot")
    return                                                                                                    



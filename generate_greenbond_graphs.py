#!/usr/local/bin/python3

# This is going to be the first plotting and regression for time series
# The tutorial for this is located at https://mohammadimranhasan.com/linear-regression-of-time-series-data-with-pandas-library-in-python/
# 1) Importing packages to work with
# First, import the package to handle the data Pandas stems from Panel Data
# it has tools for time series manipulation

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from scipy.stats import ttest_ind

import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import statsmodels.api as sm


# Second, import the package to numerical arrays this means less use of loops
# in case they are needed by using vectors and less lines of code
import numpy as np
# Scipy is a package on top of numpy for more advanced mathematical modelling
import scipy.stats as sp
# now we need somenthing for graphs and plotting
import matplotlib.pyplot as plt
# now we need to import a package to manage directories
import os
print(os.getcwd())

# 2) Setting the directory and reading the data
# first I declare a variable to call it called main_dir
main_dir = "."
os.chdir(main_dir)
# now we are going to read the data
df = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX', sheet_name='Sheet1', skiprows=1)
# I inserted a row before the data starts so that I know which bond belongs to which country
# that is why I am skipping the first row with skiprow=1
df.set_index('Date', inplace=True)
# here the first column HAS TO BE CALLED "Date" and it is case sensitive
# inplace: Makes the changes in the dataframe if True.

# 3) Viewing the data
print(df.head())
print(type(df))
print(df)

# plotting the data
df.plot()

# plotting the green bond yield as a component of the dataframe
df['CH_GREEN50_YLD_YTM_MID'].plot()

df = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX', sheet_name='Sheet1', skiprows=1)
# df[['CH_GREEN50_YLD_YTM_MID','CH_VANILLA47_YLD_YTM_MID','Bps_Chile']].plot()
df.set_index('Date', inplace=True)
df.CH_VANILLA47_YLD_YTM_MID.plot(
    grid=True, label="Vanilla", legend=True, color='orange')
df.CH_GREEN50_YLD_YTM_MID.plot(
    grid=True, label="Green", legend=True, color='blue')
df.Bps_Chile.plot(secondary_y=True, label="BPS difference", title="Chile",
                  legend=True, color="gray", ylabel='Yield', xlabel='Dates')

df = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX', sheet_name='Sheet1', skiprows=1)
# df[['CH_GREEN50_YLD_YTM_MID','CH_VANILLA47_YLD_YTM_MID','Bps_Chile']].plot()
df.set_index('Date', inplace=True)
df.FR_VANILLA41_YLD_YTM_MID.plot(grid=True, label="Vanilla", color='orange')
df.FR_GREEN39_YLD_YTM_MID.plot(grid=True, label="Green", color='blue')
plt.legend(loc='best')
df.Bps_France.plot(secondary_y=True, label="BPS difference", title="France",
                   color="gray", ylabel='Yield', xlabel='Dates')
plt.legend(loc='best')

df = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX', sheet_name='Sheet1', skiprows=1)
df.set_index('Date', inplace=True)
df.BR_VANILLA23_YLD_YTM_MID.plot(
    grid=True, label="Vanilla", color='orange', legend=True)
df.BR_GREEN24_YLD_YTM_MID.plot(
    grid=True, label="Green", legend=True, color='blue')
# plt.legend(loc='best')
df.Bps_Brazil.plot(secondary_y=True, label="BPS difference", title="Brazil BNDES", legend=True,
                   color="gray", ylabel='Yield', xlabel='Dates')
# plt.legend(loc='best')

df = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX', sheet_name='Sheet1', skiprows=1)
df.set_index('Date', inplace=True)
df.NG_VANILLA23_YLD_YTM_MID.plot(grid=True, label="Vanilla", color='orange')
df.NG_GREEN22_YLD_YTM_MID.plot(grid=True, label="Green", color='blue')
plt.legend(loc='best')
df.Bps_Nigeria.plot(secondary_y=True, label="BPS difference", title="Nigeria",
                    color="gray", ylabel='Yield', xlabel='Dates')
plt.legend(loc='lower right')

# plotting two data time series as a part of the dataframe and assigning titles and labels
df[['CH_VANILLA47_YLD_YTM_MID', 'CH_GREEN50_YLD_YTM_MID']].plot(
    title='Yield to maturity', ylabel='Basis points', xlabel='Dates', label='yields', color=['orange', 'blue'])

# dframe = df.set_index(pd.to_datetime(df.Date), drop=True)
# df.CH_VANILLA47_YLD_YTM_MID.plot(grid=True, label="bitcoin", legend=True)
# df.CH_GREEN50_YLD_YTM_MID.plot(secondary_y=True, label="tether", legend=True)

print(df.head())

# Was importing px here, should move to another file?
nuevdf = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX',
                       sheet_name='Sheet1', skiprows=1)

fig = px.line(nuevdf, x='Date', y='CH_GREEN50_YLD_YTM_MID')
fig.show()

# first I need to import plotly and assign an alias
# then I also need pandas to read the file
df2 = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX',
                    sheet_name='Sheet1', skiprows=1)
fig = px.line(df2, x='Date', y='CH_GREEN50_YLD_YTM_MID',
              title='Time Series with Rangeslider')
# could be: fig = px.line(df, x='Date', y='CH_GREEN50_YLD_YTM_MID', range_x=['2016-07-01','2016-12-31'])
fig.update_xaxes(rangeslider_visible=True)
fig.show()
# this is another example from a differen source file from the web
# import plotly.express as px
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
# fig = px.line(df, x='Date', y='AAPL.High', title='Time Series with Rangeslider')
# fig.update_xaxes(rangeslider_visible=True)
# fig.show()
# print(df)

# this is an example of multiple graphs from a differen source file from the web
# it uses a package called plotly graphs
# import plotly.graph_objs as go
# import plotly.express as px
# import pandas as pd
# Data
# gapminder = px.data.gapminder()
# Most productive european countries (as of 2007)
# df_eur = gapminder[gapminder['continent']=='Europe']
# df_eur_2007 = df_eur[df_eur['year']==2007]
# eur_gdp_top5=df_eur_2007.nlargest(5, 'gdpPercap')['country'].tolist()
# df_eur_gdp_top5 = df_eur[df_eur['country'].isin(eur_gdp_top5)]

# Most productive countries on the american continent (as of 2007)
# df_ame = gapminder[gapminder['continent']=='Americas']
# df_ame_2007 = df_ame[df_ame['year']==2007]
# df_ame_top5=df_ame_2007.nlargest(5, 'gdpPercap')['country'].tolist()
# df_ame_gdp_top5 = df_ame[df_ame['country'].isin(df_ame_top5)]

# Plotly figure 1
# fig = px.line(df_eur_gdp_top5, x='year', y='gdpPercap',
# color="country",
# line_group="country", hover_name="country")
# fig.update_layout(title='Productivity, Europe' , showlegend=False)


# Plotly figure 2
# fig2 = go.Figure(fig.add_traces(
# data=px.line(df_ame_gdp_top5, x='year', y='gdpPercap',
# color="country",
# line_group="country", line_dash='country', hover_name="country")._data))
# fig2.update_layout(title='Productivity, Europe and America', showlegend=False)

# fig.show()
# fig2.show()
# print(df_eur)
# print(df_eur_gdp_top5)

# Chilean graphs


df2 = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX',
                    sheet_name='Sheet1', skiprows=1)
fig = px.line(df2, x='Date', y=['CH_GREEN50_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID'], title='Chilean Bond Yields',
              labels={"value": "Yield", "variable": "Bonds"}, range_y=[2.6, 3.6], range_x=["2019-06-19", "2019-12-15"])
fig.update_xaxes(rangeslider_visible=True)
newnames = {'CH_GREEN50_YLD_YTM_MID': 'Green Bond 2050',
            'CH_VANILLA47_YLD_YTM_MID': 'Vanilla Bond 2047'}
fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                      legendgroup=newnames[t.name],
                                      hovertemplate=t.hovertemplate.replace(
                                          t.name, newnames[t.name])
                                      )
                   )

# fig.xlim([-10,0]) and plt.ylim([-10,0])
fig.update_layout(title={'text': "Chile", 'y': 0.9,
                  'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()

# NIGERIAN GRAPHS


df2 = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX',
                    sheet_name='Sheet1', skiprows=1)
fig = px.line(df2, x='Date', y=['NG_GREEN22_YLD_YTM_MID', 'NG_VANILLA23_YLD_YTM_MID'], title='Nigerian Bond Yields',
              labels={"value": "Yield", "variable": "Bonds"}, range_y=[9.9, 15], range_x=["2019-06-19", "2019-12-15"])
# fig.update_xaxes(rangeslider_visible=True)
newnames = {'NG_GREEN22_YLD_YTM_MID': 'Green Bond 2022',
            'NG_VANILLA23_YLD_YTM_MID': 'Vanilla Bond 2023'}
fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                      legendgroup=newnames[t.name],
                                      hovertemplate=t.hovertemplate.replace(
                                          t.name, newnames[t.name])
                                      )
                   )

# fig.xlim([-10,0]) and plt.ylim([-10,0])
fig.update_layout(title={'text': "Nigeria", 'y': 0.9,
                  'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()


# French graphs

df2 = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX',
                    sheet_name='Sheet1', skiprows=1)
fig = px.line(df2, x='Date', y=['FR_GREEN39_YLD_YTM_MID', 'FR_VANILLA41_YLD_YTM_MID'], title='French Bond Yields',
              labels={"value": "Yield", "variable": "Bonds"}, range_y=[0, .7], range_x=["2019-06-19", "2019-12-15"])
# fig.update_xaxes(rangeslider_visible=True)
newnames = {'FR_GREEN39_YLD_YTM_MID': 'Green Bond 2039',
            'FR_VANILLA41_YLD_YTM_MID': 'Vanilla Bond 2041'}
fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                      legendgroup=newnames[t.name],
                                      hovertemplate=t.hovertemplate.replace(
                                          t.name, newnames[t.name])
                                      )
                   )

# fig.xlim([-10,0]) and plt.ylim([-10,0])
fig.update_layout(title={'text': "France", 'y': 0.9,
                  'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()


# Brazilian graphs


df2 = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX',
                    sheet_name='Sheet1', skiprows=1)
fig = px.line(df2, x='Date', y=['BR_GREEN24_YLD_YTM_MID', 'BR_VANILLA23_YLD_YTM_MID'], title='Brazilian Bond Yields',
              labels={"value": "Yield", "variable": "Bonds"}, range_y=[2.8, 3.8], range_x=["2019-06-19", "2019-12-15"])
# fig.update_xaxes(rangeslider_visible=True)
newnames = {'BR_GREEN24_YLD_YTM_MID': 'Green Bond 2024',
            'BR_VANILLA23_YLD_YTM_MID': 'Vanilla Bond 2023'}
fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                      legendgroup=newnames[t.name],
                                      hovertemplate=t.hovertemplate.replace(
                                          t.name, newnames[t.name])
                                      )
                   )

# fig.xlim([-10,0]) and plt.ylim([-10,0])
fig.update_layout(title={'text': "Nigeria", 'y': 0.9,
                  'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()

# Example of the Student's t-test
# I am going to have to use a different file because the dates in the Chilean bonds one are not the same as
# in the original paper Analysis in E-views

try:
    df10 = pd.read_excel('FRANCE_BOND_SERIES.xlsx', sheet_name='Sheet1')
    df10.set_index('Date', inplace=True)
    data1 = df10['FR_Green2039']
    data2 = df10['FR_VA_2041']
    print(data1.mean())
    print(data2.mean())
    stat, p = ttest_ind(data1, data2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')
    statsSatterhwaite_Welch, probSW = ttest_ind(data1, data2, equal_var=False)
    print('statsSatterhwaite_Welch=%.3f, probSW=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

    # print(df2[['Date','FR_GREEN39_YLD_YTM_MID']])
    # selection = df2.loc[2:100,['Date','FR_GREEN39_YLD_YTM_MID']]
    # print(selection)
    # print(df2['FR_GREEN39_YLD_YTM_MID']) if (df2['Date']>10)

    # Conducting an augmented dickey-fuller test
    adfuller(data1)
    # Source https://www.statology.org/dickey-fuller-test-python/

    # Conducting a granger causality test

    # perform Granger-Causality test
    grangercausalitytests(df10[['FR_Green2039', 'FR_VA_2041']], maxlag=[7])
    grangercausalitytests(df10[['FR_VA_2041', 'FR_Green2039']], maxlag=[7])
    # Source https://www.statology.org/granger-causality-test-in-python/

    # add constant column to regress with intercept
    df11['const'] = 1
    x = df11[['FR_VA_2041', 't']]
    y = df11['FR_Green2039']
    # fit
    model = RollingOLS(endog=df11['FR_Green2039'].values, exog=df11[[
        'const', 'FR_VA_2041', 't']], window=30)
    rres = model.fit()
    # rres.params.tail(20) #look at last few intercept and coef
    print(rres)

    params = rres.params.copy()
    params.index = np.arange(1, params.shape[0] + 1)
    params.head()

    params.iloc[31:260]
    fig = rres.plot_recursive_coefficient(
        variables=["FR_VA_2041"], figsize=(14, 6))
    fig = rres.plot_recursive_coefficient(variables=["const"], figsize=(14, 6))

    # Simple regression and rolling coefficients across time for Green Bond Yields
    # I am importing statsmodels to run a univariate regression with its commands
    df11 = pd.read_excel('FRANCE_BOND_SERIES.xlsx', sheet_name='Sheet1')
    df11.set_index('Date', inplace=True)
    import statsmodels.api as sm
    x = df11[['FR_VA_2041', 't']]
    y = df11['FR_Green2039']
    # Now these are the variables
    # it is necessary to add the intercept
    x = sm.add_constant(x)
    result1 = sm.OLS(y, x).fit()
    print(result1.summary())

    # rolling coefs
    # lr = LinearRegression().fit(x, y)
    print(f'intercept = {lr.intercept_:.5f}, slope = {lr.coef_[0]:.3f}')

except Exception:
    print('Cannot read FRANCE_BOND_SERIES.xlsx')
    print(Exception)


df2 = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX',
                    sheet_name='Sheet1', skiprows=1)
fig = px.line(df2, x='Date', y=['CH_GREEN50_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID'],
              title='Chilean Bond Yields', labels={"value": "Yield", "variable": "Bonds"})
fig.update_xaxes(rangeslider_visible=True)
newnames = {'CH_GREEN50_YLD_YTM_MID': 'Green Bond 2050',
            'CH_VANILLA47_YLD_YTM_MID': 'Vanilla 2047'}
fig.for_each_trace(lambda t: t.update(name=newnames[t.name],
                                      legendgroup=newnames[t.name],
                                      hovertemplate=t.hovertemplate.replace(
                                          t.name, newnames[t.name])
                                      )
                   )
fig.update_layout(title={'text': "Chile", 'y': 0.9,
                  'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'})
fig.show()

df2.head()

# https://stackoverflow.com/questions/59762321/how-do-i-add-and-define-multiple-lines-in-a-plotly-time-series-chart
# https://plotly.com/python/time-series/
# https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea
# https://www.geeksforgeeks.org/python-pandas-dataframe-set_index/
# https://mohammadimranhasan.com/linear-regression-of-time-series-data-with-pandas-library-in-python/
# https://stackoverflow.com/questions/64371174/plotly-how-to-change-variable-label-names-for-the-legend-in-a-plotly-express-li

# I am importing statsmodels to run a univariate regression with its commands
x = df2['FR_VANILLA41_YLD_YTM_MID'].tolist()
y = df2['FR_GREEN39_YLD_YTM_MID'].tolist()
# Now these are the variables
# it is necessary to add the intercept
x = sm.add_constant(x)
result1 = sm.OLS(y, x).fit()
print(result1.summary())

# now I am going to run a multivariate regression
x = df2[['FR_VANILLA41_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID']]
print(type(x))
# here the dataframe has the "features" or independent variables in '' and uses two square brakets
y = df2['FR_GREEN39_YLD_YTM_MID']
x = sm.add_constant(x)
# this command adds the intercept
result2 = sm.OLS(y, x).fit()
print(result2.summary())

# now I am going to run a multivariate regression
x = df2[['FR_VANILLA41_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID']]
print(type(x))
# here the dataframe has the "features" or independent variables in '' and uses two square brakets
y = df2['FR_GREEN39_YLD_YTM_MID']
x = sm.add_constant(x)
print(type(y))
# this command adds the intercept
result2 = sm.OLS(y, x).fit()
print(result2.summary())

# now I am going to run a multivariate regression
x = df2[['FR_VANILLA41_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID']]
print(type(x))
# here the dataframe has the "features" or independent variables in '' and uses two square brakets
y = df2['FR_GREEN39_YLD_YTM_MID']
x = sm.add_constant(x)
# this command adds the intercept
result2 = sm.OLS(y, x).fit()
print(result2.summary())

# now I am going to add a deterministic trend variable
# t=[i+1 for i in range(0,368)]
trend = list(range(1, 368))
print(type(trend))

# x2.append((trend), ignore_index=False)
# print(type(x2))
# x2.head()
# print(x2)
# x2.insert(loc=1, column='Top Player Countries', value=trend, allow_duplicates=True)
# x2.head()
# x2=np.append(x,trend)
# print(type(x2))
# print(x2)
# print(x)
# print(result.summary())
# x3 = df2[['FR_VANILLA41_YLD_YTM_MID','CH_VANILLA47_YLD_YTM_MID']]
# print(type(x3))

# here the dataframe has the "features" or independent variables in '' and uses two square brakets
# y = df2['FR_GREEN39_YLD_YTM_MID']
# x = sm.add_constant(x)
# this command adds the intercept
# result2 = sm.OLS(y, x).fit()
# print(result2.summary())

# running the regression with Scikitlearn
# first I have to import the linear regression package
# next, I declare the variables
x = 'CH_GREEN50_YLD_YTM_MID'
y = 'CH_VANILLA47_YLD_YTM_MID'
data, target = df2[[x]], df2[y]
lr = LinearRegression()
lr.fit(data, target)


# now I am going to generate the variable trend which is a deterministic trend
# t=[x+1 for x in range(0,368)]
trend = list(range(1, 368))
print(type(x))
x2 = np.append(x, trend)
print(type(x2))
print(x2)
# print(x)
# result = sm.OLS(y, x).fit()
# print(result.summary())

# running the regression with Scikitlearn
# first I have to import the linear regression package
# next, I declare the variables
X = df[['FR_VANILLA41_YLD_YTM_MID',
        'CH_GREEN50_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID']]
Y = df['FR_GREEN39_YLD_YTM_MID']
# data, target=df2[[x]],df2[y]
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print(regr.coef_)

print(type(x))
print(df2.describe())
print(df2.info())

print(type(x2))
print(type(trend))
dftrend = pd.DataFrame(trend)
# dfcmplto = x2.append(pd.DataFrame(trend))

# x2.append(trend)
# print(trend)

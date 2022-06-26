#!/usr/local/bin/python3

# This is going to be the first plotting and regression for time series
# The tutorial for this is located at https://mohammadimranhasan.com/linear-regression-of-time-series-data-with-pandas-library-in-python/
# 1) Importing packages to work with
# First, import the package to handle the data Pandas stems from Panel Data
# it has tools for time series manipulation

from os import chdir
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.stattools import adfuller
from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import statsmodels.api as sm
import scipy.stats as sp


####################################################################
# @plotly_fig_obj px.line object - generated from a dataframe      #
# @new_name_dict  dict - { key : value } where the value will      #
#                       represent the key in the plot legend       #
# @return None - a side-effect method to update the plotly_fig_obj #
####################################################################
def sheets_to_labels(plotly_fig_obj, new_name_dict):
    plotly_fig_obj.for_each_trace(
        lambda t: t.update(name=new_name_dict[t.name],
                           legendgroup=new_name_dict[t.name],
                           hovertemplate=t.hovertemplate.replace(
                               t.name, new_name_dict[t.name])
                           )
    )


# Setting the directory and reading the data
main_dir = "."
chdir(main_dir)

# Should only read excel sheet once, unless different file.
df = pd.read_excel('CHILEAN_BONDS_EXCEL.XLSX', sheet_name='Sheet1', skiprows=1)
# I inserted a row before the data starts so that I know which bond belongs to which country
# that is why I am skipping the first row with skiprow=1
df.set_index('Date', inplace=True)
# here the first column HAS TO BE CALLED "Date" and it is case sensitive
# inplace: Makes the changes in the dataframe if True.

# plotting the data
# assign the dataframe to a variable so that additional operations don't manipulate it
df_cumulative = df

# dataframe plots are kept for reference, but don't generate browser-viewable plots
df_cumulative.plot(title="Chilean Bonds")

fig_cum = px.line(df_cumulative,
                  title='Cumulative Plot',
                  labels={
                      "value": "Yield",
                      "variable": "Bonds"})
fig_cum.update_layout(title={'text': 'Cumulative Plot',
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'})
fig_cum.show()


##################################################################

# Bps Series

df_bps = df[['Bps_Brazil', 'Bps_Chile', 'Bps_France', 'Bps_Nigeria']]
fig_bps = px.line(df_bps, title='Bps Series Comparison')
sheets_to_labels(fig_bps, {
    'Bps_Brazil': 'Brazil',
    'Bps_Chile': 'Chile',
    'Bps_France': 'France',
    'Bps_Nigeria': 'Nigeria'
})
fig_bps.update_layout(title={'text': 'Bps Series Comparison',
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'})
# rangeslider does not work with BPS series comparisons (default visibility is set to False)
fig_bps.update_xaxes(rangeslider_visible=False)
fig_bps.show()

##################################################################

# Chilean graphs

# plotting the green bond yield as a component of the dataframe
df_ch1 = df['CH_GREEN50_YLD_YTM_MID']
# dataframe plots are kept for reference, but don't generate browser-viewable plots
df_ch1.plot()

fig_ch1 = px.line(df_ch1,
                  title='Chilean Green 2050',
                  labels={
                      "value": "Yield",
                      "variable": "Bonds"})
sheets_to_labels(fig_ch1, {'CH_GREEN50_YLD_YTM_MID': 'Green 2050'})
# rangeslider does not work with BPS series comparisons
fig_ch1.update_xaxes(rangeslider_visible=True)
fig_ch1.update_layout(title={
    'text': "Chile",
    'y': 0.9,
    'x': 0.5,
    'xanchor': 'center',
    'yanchor': 'top'
})
fig_ch1.show()

##

df_ch2_bps = df[['CH_GREEN50_YLD_YTM_MID',
                 'CH_VANILLA47_YLD_YTM_MID',
                 'Bps_Chile']]
# dataframe plots are kept for reference, but don't generate browser-viewable plots
df_ch2_bps.CH_VANILLA47_YLD_YTM_MID.plot(
    grid=True, label="Vanilla", legend=True, color='orange')
df_ch2_bps.CH_GREEN50_YLD_YTM_MID.plot(
    grid=True, label="Green", legend=True, color='blue')
df_ch2_bps.Bps_Chile.plot(secondary_y=True,
                          label="BPS difference",
                          title="Chile",
                          legend=True,
                          color="gray",
                          ylabel='Yield',
                          xlabel='Dates')

# TODO: adjust plot line colors, get Bps_Chile to secondary_y
fig_ch2_bps = px.line(df_ch2_bps,
                      title='Chile',
                      labels={
                          "value": "Yield",
                          "variable": "Bonds"})
sheets_to_labels(fig_ch2_bps, {
    'CH_GREEN50_YLD_YTM_MID': 'Green 2050',
    'CH_VANILLA47_YLD_YTM_MID': 'Vanilla 2047',
    'Bps_Chile': 'Bps'
})
fig_ch2_bps.update_layout(title={
    'text': "Chile",
    'y': 0.9,
    'x': 0.5,
    'xanchor': 'center',
    'yanchor': 'top'
})
fig_ch2_bps.show()

##

# demonstrating the ability to set the initial rangeslider time series
df_ch3 = df[['CH_GREEN50_YLD_YTM_MID']]
fig_ch3 = px.line(df_ch3,
                  y='CH_GREEN50_YLD_YTM_MID',
                  range_x=['2019-06-15', '2020-11-13'],
                  title='Chile Green Bond Set Time Series with Rangeslider',
                  labels={
                      "value": "Yield",
                      "variable": "Bonds"})
# sheets_to_labels(fig8, {'CH_GREEN50_YLD_YTM_MID': 'Green 2050'})
fig_ch3.update_xaxes(rangeslider_visible=True)
fig_ch3.update_layout(title={
    'text': "Chile - Set Time Series with Rangeslider",
    'y': 0.9,
    'x': 0.5,
    'xanchor': 'center',
    'yanchor': 'top'
})
fig_ch3.show()

##

df_ch4 = df[['CH_GREEN50_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID']]
fig_ch4 = px.line(df_ch4,
                  y=['CH_GREEN50_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID'],
                  title='Chilean Bond Yields',
                  labels={
                      "value": "Yield",
                      "variable": "Bonds"},
                  range_y=[2.0, 5.0],
                  range_x=["2019-06-19", "2019-12-15"])
sheets_to_labels(fig_ch4, {
    'CH_GREEN50_YLD_YTM_MID': 'Green Bond 2050',
    'CH_VANILLA47_YLD_YTM_MID': 'Vanilla Bond 2047'
})
fig_ch4.update_layout(title={
    'text': "Chile",
    'y': 0.9,
    'x': 0.5,
    'xanchor': 'center',
    'yanchor': 'top'
})
fig_ch4.update_xaxes(rangeslider_visible=True)
fig_ch4.show()

##

df_ch5 = df[['CH_GREEN50_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID']]
fig_ch5 = px.line(df_ch5,
                  y=['CH_GREEN50_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID'],
                  title='Chilean Bond Yields',
                  labels={
                      "value": "Yield",
                      "variable": "Bonds"
                  })
sheets_to_labels(fig_ch5, {
    'CH_GREEN50_YLD_YTM_MID': 'Green Bond 2050',
    'CH_VANILLA47_YLD_YTM_MID': 'Vanilla 2047'
})
fig_ch5.update_layout(title={'text': "Chile",
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'})
fig_ch5.update_xaxes(rangeslider_visible=True)
fig_ch5.show()

##################################################################

# French graphs


df_fr = df[['FR_GREEN39_YLD_YTM_MID', 'FR_VANILLA41_YLD_YTM_MID']]
fig_fr = px.line(df_fr,
                 y=['FR_GREEN39_YLD_YTM_MID', 'FR_VANILLA41_YLD_YTM_MID'],
                 title='French Bond Yields',
                 labels={
                     "value": "Yield",
                     "variable": "Bonds"},
                 range_y=[-0.1, .7],
                 range_x=["2019-06-19", "2019-12-15"])
sheets_to_labels(fig_fr, {
    'FR_GREEN39_YLD_YTM_MID': 'Green Bond 2039',
    'FR_VANILLA41_YLD_YTM_MID': 'Vanilla Bond 2041'
})
fig_fr.update_layout(title={'text': "France",
                            'y': 0.9,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'})
fig_fr.update_xaxes(rangeslider_visible=True)
fig_fr.show()

##

df_fr_bps = df[['FR_VANILLA41_YLD_YTM_MID',
                'FR_GREEN39_YLD_YTM_MID',
                'Bps_France']]
df_fr_bps.FR_VANILLA41_YLD_YTM_MID.plot(
    grid=True,
    label="Vanilla",
    color='orange')
df_fr_bps.FR_GREEN39_YLD_YTM_MID.plot(grid=True, label="Green", color='blue')
plt.legend(loc='best')
df_fr_bps.Bps_France.plot(secondary_y=True,
                          label="BPS difference",
                          title="France",
                          color="gray",
                          xlabel='Dates',
                          ylabel='Yield')
plt.legend(loc='best')

fig_fr_bps = px.line(df_fr_bps,
                     title='France',
                     labels={
                         "value": "Yield",
                         "variable": "Bonds"})
sheets_to_labels(fig_fr_bps, {
    'FR_VANILLA41_YLD_YTM_MID': 'Vanilla 2041',
    'FR_GREEN39_YLD_YTM_MID': 'Green 2039',
    'Bps_France': 'Bps'
})
fig_fr_bps.update_layout(title={'text': "France",
                                'y': 0.9,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'})
fig_fr_bps.show()


##################################################################


# NIGERIAN GRAPHS

df_ng = df[['NG_GREEN22_YLD_YTM_MID', 'NG_VANILLA23_YLD_YTM_MID']]
fig_ng = px.line(df_ng,
                 y=['NG_GREEN22_YLD_YTM_MID', 'NG_VANILLA23_YLD_YTM_MID'],
                 title='Nigerian Bond Yields',
                 labels={
                     "value": "Yield",
                     "variable": "Bonds"},
                 range_y=[2.5, 15],
                 range_x=["2019-06-19", "2019-12-15"])
sheets_to_labels(fig_ng, {
    'NG_GREEN22_YLD_YTM_MID': 'Green Bond 2022',
    'NG_VANILLA23_YLD_YTM_MID': 'Vanilla Bond 2023'
})
fig_ng.update_layout(title={'text': "Nigeria",
                            'y': 0.9,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top'})
fig_ng.update_xaxes(rangeslider_visible=True)
fig_ng.show()

##

df_ng_bps = df[['NG_VANILLA23_YLD_YTM_MID',
                'NG_GREEN22_YLD_YTM_MID', 'Bps_Nigeria']]
df_ng_bps.NG_VANILLA23_YLD_YTM_MID.plot(
    grid=True, label="Vanilla", color='orange')
df_ng_bps.NG_GREEN22_YLD_YTM_MID.plot(grid=True, label="Green", color='blue')
plt.legend(loc='best')
df_ng_bps.Bps_Nigeria.plot(secondary_y=True, label="BPS difference", title="Nigeria",
                           color="gray", ylabel='Yield', xlabel='Dates')
plt.legend(loc='lower right')

# plotting two data time series as a part of the dataframe and assigning titles and labels
fig_ng_bps = px.line(df_ng_bps,
                     title='Nigeria',
                     labels={
                         "value": "Yield",
                         "variable": "Bonds"})
sheets_to_labels(fig_ng_bps, {
    'NG_VANILLA23_YLD_YTM_MID': 'Vanilla 2023',
    'NG_GREEN22_YLD_YTM_MID': 'Green 2022',
    'Bps_Nigeria': 'Bps'
})
fig_ng_bps.update_layout(title={'text': "Nigeria",
                                'y': 0.9,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'})
fig_ng_bps.show()


##################################################################


# Brazilian graphs

df_br_bps = df[['BR_VANILLA23_YLD_YTM_MID',
                'BR_GREEN24_YLD_YTM_MID',
               'Bps_Brazil']]
df_br_bps['BR_VANILLA23_YLD_YTM_MID'].plot(grid=True,
                                           label="Vanilla",
                                           color='orange',
                                           legend=True)
df_br_bps['BR_GREEN24_YLD_YTM_MID'].plot(grid=True,
                                         label="Green",
                                         legend=True,
                                         color='blue')
plt.legend(loc='best')

df_br_bps.Bps_Brazil.plot(secondary_y=True,
                          label="BPS difference",
                          title="Brazil BNDES",
                          legend=True,
                          color="gray",
                          ylabel='Yield',
                          xlabel='Dates')
plt.legend(loc='best')

fig_br_bps = px.line(df_br_bps,
                     title='Brazil BNDES',
                     labels={
                         "value": "Yield",
                         "variable": "Bonds"})
sheets_to_labels(fig_br_bps, {
    'BR_VANILLA23_YLD_YTM_MID': 'Vanilla 2023',
    'BR_GREEN24_YLD_YTM_MID': 'Green 2024',
    'Bps_Brazil': 'Bps'
})
fig_br_bps.update_layout(title={'text': "Brazil",
                                'y': 0.9,
                                'x': 0.5,
                                'xanchor': 'center',
                                'yanchor': 'top'})
fig_br_bps.show()

##

df_br2 = df[['BR_GREEN24_YLD_YTM_MID', 'BR_VANILLA23_YLD_YTM_MID']]
fig_br2 = px.line(df_br2,
                  y=['BR_GREEN24_YLD_YTM_MID', 'BR_VANILLA23_YLD_YTM_MID'],
                  labels={
                      "value": "Yield",
                      "variable": "Bonds"},
                  range_y=[1.6, 7.3],
                  range_x=["2019-06-19", "2019-12-15"])
sheets_to_labels(fig_br2, {
    'BR_GREEN24_YLD_YTM_MID': 'Green Bond 2024',
    'BR_VANILLA23_YLD_YTM_MID': 'Vanilla Bond 2023'
})

fig_br2.update_layout(title={'text': "Brazil",
                             'y': 0.9,
                             'x': 0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'})
fig_br2.update_xaxes(rangeslider_visible=True)
fig_br2.show()

##################################################################


# Example of the Student's t-test
# I am going to have to use a different file because the dates in the Chilean bonds one are not the same as
# in the original paper Analysis in E-views

try:
    df_fr_excel = pd.read_excel('FRANCE_BOND_SERIES.xlsx', sheet_name='Sheet1')
    df_fr_excel.set_index('Date', inplace=True)
    data1 = df_fr_excel['FR_Green2039']
    data2 = df_fr_excel['FR_VA_2041']
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
    grangercausalitytests(
        df_fr_excel[['FR_Green2039', 'FR_VA_2041']], maxlag=[7])
    grangercausalitytests(
        df_fr_excel[['FR_VA_2041', 'FR_Green2039']], maxlag=[7])
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

##################################################################

# https://stackoverflow.com/questions/59762321/how-do-i-add-and-define-multiple-lines-in-a-plotly-time-series-chart
# https://plotly.com/python/time-series/
# https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea
# https://www.geeksforgeeks.org/python-pandas-dataframe-set_index/
# https://mohammadimranhasan.com/linear-regression-of-time-series-data-with-pandas-library-in-python/
# https://stackoverflow.com/questions/64371174/plotly-how-to-change-variable-label-names-for-the-legend-in-a-plotly-express-li

# I am importing statsmodels to run a univariate regression with its commands
try:
    x = df['FR_VANILLA41_YLD_YTM_MID'].tolist()
    y = df['FR_GREEN39_YLD_YTM_MID'].tolist()
    # Now these are the variables
    # it is necessary to add the intercept
    x = sm.add_constant(x)
    result1 = sm.OLS(y, x).fit()
    print(result1.summary())

    # now I am going to run a multivariate regression
    x = df[['FR_VANILLA41_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID']]
    print(type(x))
    # here the dataframe has the "features" or independent variables in '' and uses two square brakets
    y = df['FR_GREEN39_YLD_YTM_MID']
    x = sm.add_constant(x)
    # this command adds the intercept
    result2 = sm.OLS(y, x).fit()
    print(result2.summary())

    # now I am going to run a multivariate regression
    x = df[['FR_VANILLA41_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID']]
    print(type(x))
    # here the dataframe has the "features" or independent variables in '' and uses two square brakets
    y = df['FR_GREEN39_YLD_YTM_MID']
    x = sm.add_constant(x)
    print(type(y))
    # this command adds the intercept
    result2 = sm.OLS(y, x).fit()
    print(result2.summary())

    # now I am going to run a multivariate regression
    x = df[['FR_VANILLA41_YLD_YTM_MID', 'CH_VANILLA47_YLD_YTM_MID']]
    print(type(x))
    # here the dataframe has the "features" or independent variables in '' and uses two square brakets
    y = df['FR_GREEN39_YLD_YTM_MID']
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
    data, target = df[[x]], df[y]
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
    print(df.describe())
    print(df.info())

    print(type(x2))
    print(type(trend))
    dftrend = pd.DataFrame(trend)
    # dfcmplto = x2.append(pd.DataFrame(trend))

    # x2.append(trend)
    # print(trend)
except(Exception):
    print('Linear Regression analysis not available until France Bond data included')

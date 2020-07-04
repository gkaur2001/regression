# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
"""
Created on Fri Jun 26 11:10:33 2020

@author: gurle
"""
data1 = pd.read_excel('hedge_data.xls', sheet_name = 0, index_col = 0)
data2 = pd.read_excel('hedge_data.xls', sheet_name = 1, index_col = 0)
data3 = pd.read_excel('hedge_data.xls', sheet_name = 2, index_col = 0)
data = pd.concat([data1, data2, data3], axis = 1)

##################### LTCM RISK #######################
# 1 - SUMMARY STATISTICS
def annualized_mean(data):
    return data.mean() * 12

def annualized_vol(data):
    return data.std() * np.sqrt(12)

ann_mean = annualized_mean(data1)
ann_vol = annualized_vol(data1)
summary = pd.DataFrame({'Mean': ann_mean, 'Volatility': ann_vol, 'Sharpe': ann_mean/ann_vol})
print(summary)
print()

# 2 - NET LTCM EXCESS RETURNS
res = smf.ols(formula = 'net ~ Q(\'Market Equity Index\')', data = data, missing = 'drop').fit()
alpha, beta = res.params
print('Alpha: ', alpha)
print('Beta: ', beta)
r_squared = res.rsquared
print('R-Sqaured: ', r_squared)
print()

# 3 - REGRESSION BASED METRICS
treynor_ratio = data['net'].mean() / beta
print('Treynor Ratio: ', treynor_ratio)
info_ratio = alpha/res.resid.std()
print('Information Ratio: ', info_ratio)
print()

# 4 - TAIL RISK
fifth_worst = data['net'].nsmallest(5, keep = 'first')[4]
print('Fifth worst return: ', fifth_worst)
worst_four = data['net'].nsmallest(4).mean()
print('Mean of worst four returns: ', worst_four)
skew = data['net'].skew()
print('Skew: ', skew)
kurtosis = data['net'].kurtosis()
print('Kurtosis: ', kurtosis)

################# OTHER HEDGES ##################

# 1 - SUMMARY STATISTICS
hdg_mean = annualized_mean(data2)
hdg_vol = annualized_vol(data2)
hdg_skew = data2.skew()
hdg_kurtosis = data2.kurtosis()
hdg_top_fifth = data2.quantile(0.05)
hdg_summary = pd.DataFrame({'Mean' : hdg_mean, 'Volatility' : hdg_vol, 'Skewness' : hdg_skew, 
                            'Kurtosis' : hdg_kurtosis, 'Fifth Percentile' : hdg_top_fifth})
print(hdg_summary)

# 2 - HEDGE REGRESSIONS
def calculate_reg_metrics (x_value):
    print(x_value.name)
    regression = sm.OLS(data['Market Equity Index'], sm.add_constant(x_value), missing = 'drop').fit()
    alpha, beta = regression.params
    print('Alpha: ', alpha)
    print('Beta: ', beta)
    r_squared = regression.rsquared
    print('R-Sqaured: ', r_squared)
    treynor_ratio = x_value.mean() / beta
    print('Treynor Ratio: ', treynor_ratio)
    info_ratio = alpha/regression.resid.std()
    print('Information Ratio: ', info_ratio)
    print()
    return(regression)
    
    
res_ti = calculate_reg_metrics(data['Total Index'])
res_ca = calculate_reg_metrics(data['Convertible Arbitrage'])
res_dsb = calculate_reg_metrics(data['Dedicated Short Bias'])
res_em = calculate_reg_metrics(data['Emerging Markets'])
res_emn = calculate_reg_metrics(data['Equity Market Neutral'])
res_ed = calculate_reg_metrics(data['Event Driven'])
res_edd = calculate_reg_metrics(data['Event Driven Distressed'])
res_edm = calculate_reg_metrics(data['Event Driven Multi-Strategy'])
res_edra = calculate_reg_metrics(data['Event Driven Risk Arbitrage'])
res_fia = calculate_reg_metrics(data['Fixed Income Arbitrage'])
res_gm = calculate_reg_metrics(data['Global Macro'])
res_lse = calculate_reg_metrics(data['Long/Short Equity'])
res_mf = calculate_reg_metrics(data['Managed Futures'])
res_sm = calculate_reg_metrics(data['Multi-Strategy'])






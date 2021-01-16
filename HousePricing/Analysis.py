# %%
import sklearn.datasets as sk
import sklearn.model_selection as skm
import sklearn.linear_model as skl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
boston_dataset = sk.load_boston()
# attributes of the dataset
print(dir(boston_dataset))


# %%
data = pd.DataFrame(data=boston_dataset.data,
                    columns=boston_dataset.feature_names)
# target values
data['PRICE'] = boston_dataset.target
# sample data-view and prices are stored in target attribute
print(data.head())
# check for null values by either of these methods
print(pd.isnull(data).any())
print(data.info())

# %%
# GRAPH-1
plt.figure(figsize=(10, 6))
plt.xlabel('Price in thousands')
plt.ylabel('Number of houses')
plt.hist(data['PRICE'], bins=50, ec='black', color='teal')
plt.show()

# %%
# GRAPH-2
plt.figure(figsize=(10, 6))
plt.xlabel('Number of Rooms')
plt.ylabel('Number of houses')
plt.hist(data['RM'], ec='black', color='purple')
data['RM'].mean()
plt.show()

# %%
# GRAPH-3
print(data['RAD'].value_counts())
plt.figure(figsize=(10, 6))
plt.xlabel('Accessibility of Highways')
plt.ylabel('Number of houses')
plt.hist(data['RAD'], ec='black', bins=24, color='orange')
plt.show()


# %%
data.describe()
# calculates correlation between two attributes
data['PRICE'].corr(data['RM'])
data['PRICE'].corr(data['PTRATIO'])
mask = np.zeros_like(data.corr())
triagnle_indices = np.triu_indices_from(mask)
# Correlation Matrix
mask[triagnle_indices] = True
plt.figure(figsize=(16, 10))
sns.heatmap(data.corr(), mask=mask, annot=True, annot_kws={'size': '14'})
plt.xticks(fontsize='15')
plt.yticks(fontsize='15')

# %%
# MORE GRAPHS
plt.figure(figsize=(9, 6))
plt.title('DIS vs NOX', fontsize='14')
plt.xlabel('Ditance from employment', fontsize='14')
plt.ylabel('Nitric Oxide Pollution', fontsize='14')
plt.scatter(data['DIS'], data['NOX'])
sns.set_style('darkgrid')
sns.set_context('talk')
sns.jointplot(data['NOX'], data['DIS'], size=7)
plt.show()

# %%
# COmparing correlation value and their graphs
sns.lmplot('TAX', 'RAD', data, size=7)
plt.figure(figsize=(9, 6))
plt.title('RM vs PRICE', fontsize='14')
plt.xlabel('Number of Rooms', fontsize='14')
plt.ylabel('Price of houses', fontsize='14')
plt.scatter(data['RM'], data['PRICE'])
sns.lmplot('RM', 'PRICE', data, size=7)
plt.show()

# %%
# scatter Plots for each and every attribute pair
sns.pairplot(data, kind='reg', plot_kws={'line_kws': {'color': 'cyan'}})
plt.show()

# %%
# Set features and target and build a testing and training model
prices = data['PRICE']
features = data.drop('PRICE', axis=1)
xtrain, xtest, ytrian, ytest = skm.train_test_split(
    features, prices, test_size=0.2, random_state=10)
# build intercepts and check values of coefficients
regression = skl.LinearRegression()
regression.fit(xtrain, ytrian)
pd.DataFrame(data=regression.coef_, index=xtrain.columns,
             columns=['coefficients'])

# %%
# Determining p-value for the given co-efficients
# p-values must be less than or equal to 0.05 for significance
# Calculates the constant b in y=mx+b based on data
x_const = sm.add_constant(xtrain)
model = sm.OLS(ytrian, x_const)
results = model.fit()
pd.DataFrame({'co-efficients': results.params,
              'p-value': round(results.pvalues, 3)})


# %%
# Testing for collinearity
# VIF tests for collinearity where its
# threshold must be less than 10 for significance
vif = []
for i in range(x_const.shape[1]):
    vif.append(variance_inflation_factor(exog=x_const.values, exog_idx=i))
pd.DataFrame({'Co-efficients': x_const.columns, 'VIF': np.around(vif, 2)})

# %%
# To remove features we can calculate BIC value
# if we get relatively lower value of BIC by discarding
# some features then we can permanently discard them from calculation
x_const = sm.add_constant(xtrain)
tmp = x_const
for i in xtrain.columns:
    x_const = x_const.drop(i, axis=1)
    model = sm.OLS(ytrian, x_const)
    results = model.fit()
    x_const = tmp
    print("BIC value discarding "+i + " is "+str(round(results.bic, 3)))
# attributes AGE and INDUS can be discarded


# %%
# Calculating residuals,if patterns are observed,models
# need to be improved
plt.figure(figsize=(9, 6))
prices = np.log(data['PRICE'])
features = data.drop(['PRICE', "AGE", "INDUS"], axis=1)
xtrain, xtest, ytrain, ytest = skm.train_test_split(
    features, prices, test_size=0.2, random_state=10)
x_const = sm.add_constant(xtrain)
model = sm.OLS(ytrain, x_const)
results = model.fit()
corr = round(ytrain.corr(results.fittedvalues), 2)
plt.scatter(ytrain, results.fittedvalues, alpha=0.6, c='red')
plt.xlabel('Actual log Prices')
plt.ylabel('Predicted log prices')
plt.title('Actual vs Predicted Log Prices')
plt.plot(ytrain, ytrain, color='navy')
plt.show()

# Now plotting values as residuals vs predicted values
# Pattern recognizing must be done here
plt.figure(figsize=(9, 6))
plt.scatter(results.fittedvalues, results.resid, alpha=0.6, color='red')
plt.title('Residuals vs Predicted values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Checking for Normality
# Skewness must be close to zero
sns.distplot(results.resid, color='navy')
plt.title('Residuals (Skewness : '+str(round(results.resid.skew(), 3))+")")
plt.show()

log_residuals = results.rsquared
print(log_residuals)
log_mse = results.mse_resid


# %%
# Plots using All features i.e. Original Model
plt.figure(figsize=(9, 6))
prices = data['PRICE']
features = data.drop('PRICE', axis=1)
xtrain, xtest, ytrain, ytest = skm.train_test_split(
    features, prices, test_size=0.2, random_state=10)
x_const = sm.add_constant(xtrain)
model = sm.OLS(ytrain, x_const)
results = model.fit()
corr = round(ytrain.corr(results.fittedvalues), 2)
plt.scatter(ytrain, results.fittedvalues, alpha=0.6, c='gold')
plt.xlabel('Actual log Prices')
plt.ylabel('Predicted log prices')
plt.title('Actual vs Predicted Log Prices')
plt.plot(ytrain, ytrain, color='navy')
plt.show()

# Now plotting values as residuals vs predicted values
# Pattern recognizing must be done here
plt.figure(figsize=(9, 6))
plt.scatter(results.fittedvalues, results.resid, alpha=0.6, color='gold')
plt.title('Residuals vs Predicted values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Checking for Normality
# Skewness must be close to zero
sns.distplot(results.resid, color='indigo')
plt.title('Residuals (Skewness : '+str(round(results.resid.skew(), 3))+")")
plt.show()

full_residuals = results.rsquared
full_mse = results.mse_resid

# %%
# Calculating MSE and R-Squred values
# R-Squared gives percentage propotion of variance in data
# Normal distribution covers 68% data between 1SD and -1SD
# Normal distribution covers 95% data between 2SD abd -2SD which
# gives us the range of predicted value
pd.DataFrame({'R-Squared': [log_residuals, full_residuals],
              'MSE': [log_mse, full_mse],
              'RMSE': np.sqrt([log_mse, full_mse]),
              }, index=['Reduced Log Model', 'Full Price Model'])

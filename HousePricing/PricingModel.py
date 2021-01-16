# %%
# Loading required Modules
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# %%
# Loading the complete Dataset
boston_dataset = load_boston()
# Building and arranging Data
data = pd.DataFrame(data=boston_dataset.data,
                    columns=boston_dataset.feature_names)
# Optimized Version of the data used in analysis
features = data.drop(['INDUS', 'AGE'], axis=1)
# Taking log of the target values to reduce the skewness
log_prices = np.log(boston_dataset.target)
# Bulding and arranging the Price Dataset
target = pd.DataFrame(log_prices, columns=['PRICE'])
features.head()


# %%
# Create an empty array to replicate the features
property_stats = np.ndarray(shape=(1, 11))
# First row of values will contain the mean values of each feature
property_stats = features.mean().values.reshape(1, 11)


# %%
# Performing regression and obtaining mse and rmse
regression = LinearRegression().fit(features, target)
fitted_values = regression.predict(features)
mse = mean_squared_error(target, fitted_values)
rmse = np.sqrt(mse)

# %%
# Perform Prediction based on mean/user inputted values
# Indices based on the Features table
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8


def logEstimate(nr_rooms, ptRatio, river=False, confidence=False):
    # Predict values based on user input if provided
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = ptRatio
    property_stats[0][RM_IDX] = 1 if river else 0
    estimate = regression.predict(property_stats)
    if confidence:
        upper_bound = estimate+2*rmse
        lower_bound = estimate-2*rmse
        interval = 95
    else:
        upper_bound = estimate+rmse
        lower_bound = estimate-rmse
        interval = 68
    return estimate, upper_bound, lower_bound, interval

# %%


def getEstimate(nr_rooms, ptRatio, river=False, confidence=False):
    """
    Estimate a price of Property in Boston.

    Keyword Arguments:

    nr_rooms -- Number of rooms in the property

    PTRATIO -- Number of students per teacher in classroom for nearby schools

    CHAS -- True if property is near the river else 0 

    Confidence -- True if prediction range is 95% accurate or False for 68% accuracy

    """
    # COndition checking
    if(nr_rooms < 1 or nr_rooms > 9):
        print('Invalid Value of Rooms,it cannot exceed 9')
        return
    elif(ptRatio < 1 or ptRatio > 22):
        print("Invalid value for PTRATIO,it cannot exceed 22")
        return
    # By scaling the median we can get the idea of current house Prices
    zillow_current_median = 653.6
    target_medain = np.median(boston_dataset.target)
    scale_factor = zillow_current_median/target_medain
    estimate, upper_bound, lower_bound, interval = logEstimate(
        nr_rooms, ptRatio, river, confidence)
    # Converting log based values to 1000's dollar and scaling them
    estimate = np.around(np.e**estimate*1000*scale_factor, -3)[0][0]
    upper_bound = np.around(np.e**upper_bound*1000*scale_factor, -3)[0][0]
    lower_bound = np.around(np.e**lower_bound*1000*scale_factor, -3)[0][0]
    # Printing the values
    print(f'Estimated Price of the House is : {estimate} $')
    print(
        f'The Range of the Prices can be from {lower_bound} $ to {upper_bound} $')
    print(f'Confidence interval is {interval}%')

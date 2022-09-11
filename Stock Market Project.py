# For this project, we will create a machine learning model to help us predict the stock market. With the extreme
# volatility of the market these days (thanks to the rate hiking environment created by the Federal Open Market
# Committee) stocks have plummeted, rallied, and plummeted and rallied again. Any sort of insight into the trends of
# the market this year are invaluable. As such, let's see if we can help an investor's portfolio.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Adjust our display settings
pd.set_option('display.max_columns', 12)  # This allows us to view all the column within our PyCharm IDE
pd.set_option('display.width', 1000)

# First, we specify the url of the dataset that we will be working with
path = r'https://raw.githubusercontent.com/nntrongnghia/Dataquest-Guided-Project-Predicting-the-stock-market/master/sphist.csv'

# We then create a dataframe for our dataset
markets = pd.read_csv(path)

# We then do some initial preprocessing and cleaning. We first convert the time column to a datetime object.
markets["Date"] = pd.to_datetime(markets["Date"])

# Before we apply our mask, let's sort the initial dataframe.
markets.sort_values(ascending=True, by=['Date'], inplace=True)
markets = markets.reset_index(drop=True)

# However, before we split our dataframe into the train and test sets, it's crucial to note the time series nature of
# our dataset. Each row is not independent. As such, it would be better to create a few rolling-average columns. We will
# create columns that contain data for the 5 and 30 day rolling average along with a 5-day standard deviation. I later
# added rolling volume averages to try and improve our accuracy
markets["5_day_avg"] = markets["Close"].rolling(5).mean()
markets["30_day_avg"] = markets["Close"].rolling(30).mean()
markets["5_day_std"] = markets["Close"].rolling(5).std()
markets["5_day_vol"] = markets["Volume"].rolling(5).mean()
markets["253_day_vol"] = markets["Volume"].rolling(253).mean()  # The stock market is only open 253 days per year

# Note that we have to consider a very important contingency. When computing rolling statistics for our model, we must
# shift all of our results forward by 1. This is because the rolling mean uses the current day's price. This imputes
# future knowledge into our model and will make it less accurate when executed in the real-world.
markets["5_day_avg"] = markets["5_day_avg"].shift(1, axis=0)
markets["30_day_avg"] = markets["30_day_avg"].shift(1, axis=0)
markets["5_day_std"] = markets["5_day_std"].shift(1, axis=0)
markets["5_day_vol"] = markets["5_day_vol"].shift(1, axis=0)
markets["253_day_vol"] = markets["253_day_vol"].shift(1, axis=0)

# Let's display our dataframe to understand what our data looks like
print(markets.tail())
print("\n")

# The first few rows in our dataframe have NaN values since there is no data prior to 1950. Let's just drop all data
# from 1950 and begin our analysis/model construction from 1951 and onward.
mask3 = markets["Date"] >= datetime(year=1951, month=1, day=3)
better_markets = markets[mask3]
better_markets = better_markets.dropna(axis=0)  # Drop rows with NaN values

# We wish to split our dataset on the time frame. Our data has historical information from 1950 to 2015.
# We will use the data from 1950-2012 to train our model and test it on the data from 2013 to 2015.
mask1 = better_markets["Date"] >= datetime(year=2013, month=1, day=1)
mask2 = better_markets["Date"] < datetime(year=2013, month=1, day=1)

# We are now ready to split our dataframe into a train and test set
train = better_markets[mask2]
test = better_markets[mask1]

# =========================================== MODEL CONSTRUCTION ===================================================== #
lin_reg = LinearRegression()  # Creates an instance of the Linear Regression class
X_train = train[['5_day_avg', '30_day_avg', '5_day_std', '5_day_vol', '253_day_vol']]  # Define training feature matrix
y_train = train["Close"]  # Define our training target
X_test = test[['5_day_avg', '30_day_avg', '5_day_std', '5_day_vol', '253_day_vol']]  # Define our test feature matrix
y_test = test["Close"]  # Define our test target

# We now fit our model to the training data and test it on our test set to calculate the RMSE score
lin_reg.fit(X_train, y_train)
predictions = lin_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("Our model's rmse score:", rmse)
print("\n")

# ========================================== VISUALIZATION OF MODEL ================================================== #
plt.plot(test["Date"], test["Close"], label='Actual')
plt.plot(test["Date"], predictions, label='Predicted')
plt.title("Accuracy of the Stock Market Model")
plt.xlabel("Dates")
plt.ylabel("S&P500 Index")
plt.legend()
plt.show()

# =========================================== CONCLUSION ============================================================= #
# For this project, we were able to utilize Linear Regression to help predict the S&P500 Index from 2013 to the end of
# 2016. We used historical data from 1951 to the end of 2012 to train our model. Our final RMSE score was just over 22,
# implying that our model was, on average, 22 points off in its predictions each day. Considering the fact that the
# markets swings hundreds of points in a day, sometimes thousands, this score is very accurate.

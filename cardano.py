# %% [markdown]
# Import the neccesaary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import warnings

# %% [markdown]
# Data Loading

# %%
try:
    # Load the dataset from a CSV file
    df = pd.read_csv("coin_Cardano.csv")
    print("Data loaded successfully")
except FileNotFoundError:
    print("Error: 'coin_Cardano.csv' not found. Please make sure the file is in the same directory")
    exit()
df

# %% [markdown]
# Data Preprocessing

# %%
# Rename the columns for clarity and consistency
df.rename(columns={
    "SNo":"serial_number",
    "Name":"name",
    "Symbol":"symbol",
    "Date":"date",
    "High":"high",
    "Low":"low",
    "Open":"open",
    "Close":"close",
    "Volume":"volume",
    "Marketcap":"marketcap"
},inplace=True)

# Convert the "date" column to a datetime object for time series plotting
df["date"] = pd.to_datetime(df["date"])
print(df.info())

# Check for missing values
df_missing = df.isnull().sum()
print("Missing Values")
print(df_missing)

# Check for duplicated rows
df_duplicated = df.duplicated().sum()
print("Duplicated Rows")
print(df_duplicated)

# %% [markdown]
# Features Engineering

# %%
# Define the features (X) and the target (y)

features = ["high","low","open","volume"]
target = "close"

X = df[features]
y = df[target]

# %% [markdown]
# Data Splitting

# %%
# Split the data into training and testing sets. We use 80% of the data
# for training and 20% for testing.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# %% [markdown]
# Visualization before training

# %%
plt.figure(figsize=(12,6))
plt.plot(df["date"],df["close"],label="Close Price",color="green")
plt.title("Cardano Close Price Over  Time")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Create a scatter plot to visualize the relationship between the "high" and "close" price
plt.figure(figsize=(12,6))
plt.scatter(df["high"],df["close"],color="green")
plt.title("High Price vs Close Price")
plt.xlabel("High Price (USD)")
plt.ylabel("Close Price (USD)")
plt.grid(True)
plt.show()

# %%
# Create a scatter plot to visualize the relationship between the "low" and "close" price
plt.figure(figsize=(12,6))
plt.scatter(df["low"],df["close"],color="green")
plt.title("Low Price vs Close Price")
plt.xlabel("Low Price (USD)")
plt.ylabel("Close Price (USD)")
plt.grid(True)
plt.show()

# %%
# Create a scatter plot to visualize the relationship between the "open" and "close" price
plt.figure(figsize=(12,6))
plt.scatter(df["open"],df["close"],color="green")
plt.title("Open Price vs Close Price")
plt.xlabel("Open Price (USD)")
plt.ylabel("Close Price (USD)")
plt.grid(True)
plt.show()

# %%
# Create a scatter plot to visualize the relationship between the "volume" and "close" price
plt.figure(figsize=(12,6))
plt.scatter(df["volume"],df["close"],color="green")
plt.title("Volume vs Close Price")
plt.xlabel("Volume Price (USD)")
plt.ylabel("Close Price (USD)")
plt.grid(True)
plt.show()

# %% [markdown]
# Model Training

# %%
# Initialize and train four different regression models

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)

# Ridge Regression 
ridge_reg = Ridge()
ridge_reg.fit(X_train,y_train)

# Lasso Regression
lasso_reg = Lasso()
lasso_reg.fit(X_train,y_train)

# ElasticNet Regression
elastic_reg = ElasticNet()
elastic_reg.fit(X_train,y_train)

# Support Vector Regression
svr_reg = SVR()
svr_reg.fit(X_train,y_train)

# Random forest Regressor
rfr_reg = RandomForestRegressor(n_estimators=100,random_state=42)
rfr_reg.fit(X_train,y_train)

# Make the predictions on the test set of each model
y_pred_lin = lin_reg.predict(X_test)
y_pred_ridge = ridge_reg.predict(X_test)
y_pred_lasso = lasso_reg.predict(X_test)
y_pred_elastic = elastic_reg.predict(X_test)
y_pred_svr = svr_reg.predict(X_test)
y_pred_rfr = rfr_reg.predict(X_test)

# %% [markdown]
# Model Evaluation

# %%
# Linear Regression Metrics
print("-----Linear Regression-----")
print(f"R-squared: {r2_score(y_test,y_pred_lin):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred_lin):.4f}")
print(f"MSE: {mean_squared_error(y_test,y_pred_lin):.4f}")

# Ridge Regression Metrics
print("-----Ridge Regression-----")
print(f"R-squared: {r2_score(y_test,y_pred_ridge):.4f}")
print(f"MAE: {r2_score(y_test,y_pred_ridge):.4f}")
print(f"MSE: {mean_absolute_error(y_test,y_pred_ridge):.4f}")

# Lasso Regression Metrics
print("-----Lasso Regression-----")
print(f"R-squared: {r2_score(y_test,y_pred_lasso):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred_lasso):.4f}")
print(f"MSE: {mean_squared_error(y_test,y_pred_lasso):.4f}")

# ElasticNet Regression Metrics
print("-----Elastic Regression-----")
print(f"R-squared: {r2_score(y_test,y_pred_elastic):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred_elastic):.4f}")
print(f"MSE: {mean_squared_error(y_test,y_pred_elastic):.4f}")

# Support Vector Regression
print("-----Support Vector Regression-----")
print(f"R-squared: {r2_score(y_test,y_pred_svr):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred_svr):.4f}")
print(f"MSE: {mean_squared_error(y_test,y_pred_svr):.4f}")

# Random Forest Regression
print("-----Random Forest Regression-----")
print(f"R-squared: {r2_score(y_test,y_pred_rfr):.4f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred_rfr):.4f}")
print(f"MSE: {mean_squared_error(y_test,y_pred_rfr):.4f}")

# %% [markdown]
# Determine the best model

# %%
# Determine the best model based on R-squared (or other metrics)
# In this case, R-squared is a good indicator of overall fit
r2_scores = {
    "Linear": r2_score(y_test,y_pred_lin),
    "Ridge": r2_score(y_test,y_pred_ridge),
    "Lasso": r2_score(y_test,y_pred_lasso),
    "ElasticNet": r2_score(y_test,y_pred_elastic),
    "SVR": r2_score(y_test,y_pred_svr),
    "RFR":r2_score(y_test,y_pred_rfr)
}

best_model_name = max(r2_scores,key=r2_scores.get)
print(f" Conclusion: The best performing model is {best_model_name} Regression")

# Select thr best model's predictions for the final visualization
if best_model_name == "Linear":
    y_pred_best = y_pred_lin
elif best_model_name == "Ridge":
    y_pred_best = y_pred_ridge
elif best_model_name == "Lasso":
    y_pred_best = y_pred_lasso
elif best_model_name == "ElasticNet":
    y_pred_best = y_pred_elastic
elif best_model_name == "SVR":
    y_pred_best = y_pred_svr
else:
    y_pred_best = y_pred_rfr

# %% [markdown]
# Visualization after training

# %%
# Create a visualization to compare the actual values with predictions from the best model
# This plot shows how closely the model's prediction align with real data
plt.figure(figsize=(14,7))
plt.scatter(range(len(y_test)),y_test,color="green",label="Actual Prices")
plt.scatter(range(len(y_pred_best)),y_pred_best,color="red",label="Predicted Prices")
plt.title(f"Actual vs Predicted Prices ({best_model_name}) Regression")
plt.xlabel("Test Sample Index")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# User Input and Prediction

# %%
print("-----Predict Cardano Closing Price-----")
print("Enter the following data to predict the closing price")

try:
    # Prompt the user for the faetures values
    high_price = float(input("Enter the High Price:"))
    low_price = float(input("Enter the Low Price:"))
    open_price = float(input("Enter the Open Price:"))
    volume = float(input("Enter the Volume"))

    # Create a new DataFrame with user's input
    # The data must be in the same format as the training data
    new_data = pd.DataFrame([[high_price,low_price,open_price,volume]],columns=features)

    # Use the best-performing model to make a preditcion on the new data
    predicted_price = lin_reg.predict(new_data)

    # Print the final predicted price
    print(f"Predicted Closing Price is: ${predicted_price[0]:.2f}")

except ValueError:
    print("Invalid input. Please enter valid numerical values.")
except Exception as e:
    print(f"An error occurred: {e}")



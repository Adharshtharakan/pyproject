#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf # Helper for Module 1: Importing data

# 1. Data Import (Module 1)
data = yf.download('RELIANCE.NS', start='2020-01-01', end='2024-01-01')

# 2. Data Cleaning (Module 2)
# Identifying and handling missing values 
data.dropna(inplace=True) 

# Basic insights (Module 1)
print(data.describe())


# In[2]:


# Calculating moving averages (Summarization - Module 3)
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()

# Identify Correlation (Module 3)
correlation = data['SMA50'].corr(data['SMA200'])
print(f"Correlation between short and long term averages: {correlation}")


# In[3]:


# 3. Generating Signals (Module 4: Decision-making)
data['Signal'] = 0
# Golden Cross: Buy (1) | Death Cross: Sell (-1)
data['Signal'][50:] = np.where(data['SMA50'][50:] > data['SMA200'][50:], 1, 0)
data['Position'] = data['Signal'].diff()

# 4. Visualization (CO-4)
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price', alpha=0.5)
plt.plot(data['SMA50'], label='50-day SMA')
plt.plot(data['SMA200'], label='200-day SMA')
plt.title('Reliance Industries: Golden/Death Cross Analysis')
plt.legend()
plt.show() # Visual evaluation of the 'model'


# In[4]:


# Use .loc to explicitly tell pandas which rows and column to modify
data.loc[data.index[50:], 'Signal'] = np.where(
    data['SMA50'][50:] > data['SMA200'][50:], 1, 0
)


# In[5]:


# Example of 'Refining models' (Module 5)
short_windows = [20, 40, 50]
long_windows = [100, 150, 200]

for s in short_windows:
    for l in long_windows:
        # Calculate returns for each pair to find the 'Optimal' model
        # This demonstrates 'Grid search for model optimization' (Module 5)
        pass


# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# ==========================================
# MODULE 1: Introduction to Data Analytics
# Goal: Importing and basic insights (CO-1)
# ==========================================
ticker = 'RELIANCE.NS'
# Importing data in Python 
data = yf.download(ticker, start='2020-01-01', end='2024-01-01')

print("--- Basic Insights ---")
print(data.head()) # Exploratory analysis 

# ==========================================
# MODULE 2: Data Cleaning and Preparation
# Goal: Handling missing values & formatting (CO-1, CO-3)
# ==========================================
# Identifying and handling missing values 
data = data.ffill() # Forward fill for market holidays

# Normalization/Standardization concept 
# (Scaling volume for better visualization)
data['Scaled_Volume'] = (data['Volume'] - data['Volume'].min()) / (data['Volume'].max() - data['Volume'].min())

# ==========================================
# MODULE 3: Data Summarization & Stats
# Goal: Descriptive statistics & Correlation (CO-2, CO-3)
# ==========================================
# Calculating Mean (SMA) for business insights 
data['SMA50'] = data['Close'].rolling(window=50).mean()
data['SMA200'] = data['Close'].rolling(window=200).mean()

# Correlation analysis (Module 3 Topic) 
correlation = data['SMA50'].corr(data['SMA200'])
print(f"\nCorrelation between 50-day and 200-day SMA: {correlation:.2f}")

# ==========================================
# MODULE 4: Model Development (The Algorithm)
# Goal: Decision-making based on predictions (CO-4)
# ==========================================
# Logic: Simple Regression-style decision making 
data['Signal'] = 0.0
# Using .loc to avoid the SettingWithCopyWarning
data.iloc[200:, data.columns.get_loc('Signal')] = np.where(
    data['SMA50'][200:] > data['SMA200'][200:], 1.0, 0.0
)

# Visual evaluation of the 'model' (CO-4) 
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price', alpha=0.4)
plt.plot(data['SMA50'], label='50-day SMA', color='orange')
plt.plot(data['SMA200'], label='200-day SMA', color='blue')
plt.title(f'Algorithmic Trading Signals: {ticker}')
plt.legend()
plt.show()

# ==========================================
# MODULE 5: Advanced Evaluation & Optimization
# Goal: Refining models for accuracy (CO-2)
# ==========================================
# Calculate Daily Returns
data['Daily_Return'] = data['Close'].pct_change()
# Strategy returns: If signal is 1, we get the return of the next day
data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']

# Performance metrics: MSE/R-squared logic applied to returns 
total_return = (1 + data['Strategy_Return']).prod() - 1
print(f"\n--- Model Evaluation ---")
print(f"Total Strategy Return: {total_return * 100:.2f}%")


# In[ ]:





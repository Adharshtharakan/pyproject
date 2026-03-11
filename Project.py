#!/usr/bin/env python
# coding: utf-8

# In[8]:


from scipy import stats

# Performing One-Way ANOVA (Module 3 Topic)
# Are the returns of Reliance, TCS, HDFC, and Infy significantly different?
f_stat, p_val = stats.f_oneway(returns['RELIANCE.NS'].dropna(), 
                               returns['TCS.NS'].dropna(), 
                               returns['HDFCBANK.NS'].dropna(), 
                               returns['INFY.NS'].dropna())

print(f"\n--- Statistical Validation (Module 3) ---")
print(f"ANOVA F-statistic: {f_stat:.4f}")
print(f"P-value: {p_val:.4f}")

if p_val < 0.05:
    print("Conclusion: There is a significant difference between stock returns.")
else:
    print("Conclusion: Stock returns are statistically similar.")


# In[9]:


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

# =================================================================
# MODULE 1 & 2: Data Acquisition & Preparation (CO-1, CO-3)
# Goal: Importing datasets and handling missing values
# =================================================================
# 1. Importing datasets in Python (Module 1)
tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS']
raw_data = yf.download(tickers, start='2022-01-01', end='2024-01-01')['Close']

# 2. Handling missing values for business continuity (Module 2)
data = raw_data.ffill().dropna() 

# =================================================================
# MODULE 3: Data Summarization & Statistical Analysis (CO-2, CO-3)
# Goal: Mean, Variance, and Correlation for business insights
# =================================================================
# Calculate Daily Log Returns
returns = np.log(data / data.shift(1))

# Annualized Mean and Covariance (Risk Matrix)
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# Statistical Validation: Correlation (Module 3 Topic)
print("--- Module 3: Correlation Analysis ---")
print(returns.corr()) 

# =================================================================
# MODULE 5: Advanced Model Evaluation & Optimization (CO-2)
# Goal: Grid search (Simulation) to refine model accuracy
# =================================================================
results = []
# Simulating 5,000 different investment "weights"
for _ in range(5000):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)

    p_return = np.sum(mean_returns * weights)
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Evaluating reward-to-risk (Sharpe Ratio)
    results.append([p_return, p_volatility, p_return/p_volatility, weights])

# Convert to DataFrame for analysis (Module 1)
results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe_Ratio', 'Weights'])

# Find the best business scenario (Module 5)
best_portfolio_idx = results_df['Sharpe_Ratio'].idxmax()
best_portfolio = results_df.iloc[best_portfolio_idx]

# =================================================================
# MODULE 4: Model Development & Visual Evaluation (CO-4)
# Goal: Visualizations to support decision-making
# =================================================================
print(f"\n--- Module 5: Optimized Business Recommendation ---")
print(f"Optimal Return: {best_portfolio['Return']*100:.2f}%")
print(f"Optimal Risk: {best_portfolio['Volatility']*100:.2f}%")

# Plotting the Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe_Ratio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(best_portfolio['Volatility'], best_portfolio['Return'], color='red', marker='*', s=200, label='Optimal Portfolio')

plt.title('Efficient Frontier: Risk vs Return Optimization')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Expected Return')
plt.legend()
plt.show()


# In[10]:


import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats

# --- MODULE 1 & 2: Data Acquisition & Cleaning (CO-1, CO-3) ---
tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS']
raw_data = yf.download(tickers, start='2022-01-01', end='2024-01-01')['Close']
data = raw_data.ffill().dropna() # Handling missing values (Module 2)

# --- MODULE 3: Statistical Summarization (CO-2, CO-3) ---
returns = np.log(data / data.shift(1))
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# --- MODULE 5: Optimization & Refinement (CO-2) ---
results = []
for _ in range(5000):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)
    p_return = np.sum(mean_returns * weights)
    p_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    results.append([p_return, p_volatility, p_return/p_volatility, weights])

results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe_Ratio', 'Weights'])
best_portfolio_idx = results_df['Sharpe_Ratio'].idxmax()
best_portfolio = results_df.iloc[best_portfolio_idx]

# --- EXTRACTING THE FINAL BUSINESS RECOMMENDATION ---
# Map the optimized weights back to the stock names
optimized_weights = dict(zip(tickers, best_portfolio['Weights']))

print(f"--- FINAL BUSINESS RECOMMENDATION (Module 5) ---")
print(f"To achieve the Optimal Return of {best_portfolio['Return']*100:.2f}%, invest as follows:")
for stock, weight in optimized_weights.items():
    print(f" * {stock}: {weight*100:.2f}% of total capital")

# --- MODULE 4: Visual Evaluation (CO-4) ---
plt.figure(figsize=(10, 6))
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe_Ratio'], cmap='viridis', alpha=0.3)
plt.scatter(best_portfolio['Volatility'], best_portfolio['Return'], color='red', marker='*', s=250, label='Optimal Portfolio')
plt.title('Efficient Frontier: Final Investment Strategy')
plt.xlabel('Risk (Volatility)')
plt.ylabel('Expected Return')
plt.legend()
plt.show()


# In[ ]:





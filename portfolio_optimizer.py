# IMPORT ALL THE REQUIRED MODULES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation , get_latest_prices

plt.style.use('fivethirtyeight')

# LIST OUT ALL THE instuments 
instuments = []
while (True):
    a = '-'.join(y for y in instuments )
    print(a)
    x = str(input('Enter the prefered instuments(q to quit) -> ')).upper()
    if x=='Q':
        break
    instuments.append(x)

# SET THE START DATE AND END DATE


start_date = '2012-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')


# CREATE A TEMPORARY DATAFRAME TO STORE CLOSING PRICES OF THE GIVEN instuments

df = pd.DataFrame()
for instument in instuments:
  df[instument] = yf.download(instument, start=start_date, end=end_date).Close
  
  
# PLOT A GRAPH OF instuments HISTORICAL PRICES

plt.figure(figsize=(10,5))
for instument in instuments:
  plt.plot(df[instument] , label=instument)
plt.title('Historical Price')
plt.xlabel('Date' , fontsize=20)
plt.ylabel('Close Price' , fontsize=20)
plt.legend(instuments , loc='upper left')
plt.show()
  
  
# USING PyPortfolioOpt TO GET THE OPTIMAL WEIGHTS FOR THE GIVEN INSTUMENTS
  
x = expected_returns.mean_historical_return(df)
s = risk_models.sample_cov(df)

ef = EfficientFrontier(x,s)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
text = "OPTIMAL STOCKS PERCENTAGES"
print(f'\n {text:-^70} ')
print(cleaned_weights)
print(f'\n {"PERFORMANCE":-^70} ')
ef.portfolio_performance(verbose=True)

latest_prices = get_latest_prices(df)
weights = cleaned_weights

portfolio_size = 20000
print(f'\n {"PORTFOLIO":-^70} ')
da = DiscreteAllocation(weights , latest_prices , total_portfolio_value=portfolio_size)
allocation , leftover = da.lp_portfolio()
print(f' Portfolio value -> {portfolio_size}(USD) ')
print(f' Stock allocation -> {allocation} ')
print(f' Remaining Fund -> {round(leftover,2)} ')

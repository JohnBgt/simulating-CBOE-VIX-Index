

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_rows", None, "display.max_columns", None)


# Assumption: Simulation practiced on Friday 31st of March 2023 at 16:00 ET
# Near-Term options: “standard” SPX options expiring on April 21st, 2023 (third Friday) at 9:30 A.M.
# Next-Term Options: P.M.-settled SPX Weekly’s expiring on the following week (April 28, 2023) at 4:00 P.M.


# Expressing Time to maturities of Near and Next term options with minutes as arguments
T1 = (21*24*60 + (24-16)*60 + 9*60 + 30) / (365*24*60)           # 21 days, 8 hours left before April 21st, 2023 at 00:00 + 9 hours and 30 minutes before market opening and option expiration
T2 = (28*24*60 + (24-16)*60 + 16*60) / (365*24*60)               # 28 days, 8 hours left before April 28, 2023 at 00:00 + 16 hours before market close and option expiration

R1 = 0.047755  #  risk-free interest rate for 22 days maturity using Cubic Spline Extrapolation
R2 = 0.047852  #  risk-free interest rate for 29 days maturity using Cubic Spline Extrapolation

print(f'T1 equals to {T1:.5f} years')
print(f'T2 equals to {T2:.5f} years')

Rs = [R1, R2]                              # Risk-free rates
Ts = [T1, T2]                              # Time to maturity in years
S0 = 4110                                  # Initial Stock Price or last SPX price at 16:00 ET:4109.31
Ks = np.arange(3300, 5000, 5)              # Strike Prices
r = 0.025                                  # Mean Annual risk-free rate
mkt = 0.12                                 # Annual average market return
i = 0.035                                  # Inflation rate
q = 0.015                                  # Dividend Yield
sigma = 0.15                               # Annual Mean Volatility
steps = 528                                # Time steps - 528 to get hourly data (22 days * 24 hours)

mu = mkt - r - i - q                       # Drift


# Function for Monte-Carlo Simulations of Index prices using a Geometric Brownian Motion
def sim_index_prices(S0, T, mu, sigma, steps, N):
    dt = T / steps
    Wt = np.sqrt(dt) * np.random.normal(size=(steps, N))          # Wiener Process

    #  Geometric Brownian Motion of the form S(t) = S(0) * e^((μ - σ^2/2) * t + σ * W(t))
    log_ST = np.log(S0) + np.cumsum(((mu-0.5*sigma**2)*dt + sigma*Wt), axis=0)
    St = np.exp(log_ST)
    return St


# Plot a Simulation of Index prices with low number of trials
N = 50     # Number of trials
index = sim_index_prices(S0, Ts[1], mu, sigma, steps, N)
plt.plot(index)
plt.xlabel("Time Increments: Hours")
plt.ylabel("Index Price")
plt.title("Simulation of the S&P500 index with a 22-day GBM model, N =50")
plt.show()


# Simulation of index prices in order to generate option prices for both near and next terms options.
min_abs_diffs = []
call_prices_all = []
put_prices_all = []
ATM_strikes = []
ATM_calls = []
ATM_puts = []

for T in Ts:
    N = 100000
    index = sim_index_prices(S0, T, mu,sigma, steps, N)

    call_prices = []
    put_prices = []
    for K in Ks:
        call_payoffs = np.maximum(index[-1]-K, 0)
        call_price = np.exp(-r*T)*np.mean(call_payoffs)
        call_prices.append(call_price)
        put_payoffs = np.maximum(K-index[-1], 0)
        put_price = np.exp(-r*T)*np.mean(put_payoffs)
        put_prices.append(put_price)

    call_prices_all.append(call_prices)
    put_prices_all.append(put_prices)

    diff = np.array(call_prices) - np.array(put_prices)
    min_abs_diff = np.argmin(abs(diff))
    min_abs_diffs.append(min_abs_diff)

    ATM_strike = Ks[min_abs_diff]
    ATM_call = call_prices[min_abs_diff]
    ATM_put = put_prices[min_abs_diff]
    ATM_strikes.append(ATM_strike)
    ATM_calls.append(ATM_call)
    ATM_puts.append(ATM_put)

call_prices_df = pd.DataFrame(call_prices_all).T
put_prices_df = pd.DataFrame(put_prices_all).T

Ks_df = pd.DataFrame(Ks)

Table_Near = pd.concat([Ks_df, call_prices_df[0], put_prices_df[0]], axis=1)
Table_Near.columns = ['Strike', 'Calls', 'Puts']
Table_Next = pd.concat([Ks_df, call_prices_df[1], put_prices_df[1]], axis=1)
Table_Next.columns = ['Strike', 'Calls', 'Puts']
print('Near-Term Options Table:')
print(Table_Near)
print('Next-Term Options Table:')
print(Table_Near)

Table_Near.to_csv('Near_term_Options.csv')
Table_Next.to_csv('Next_term_Options.csv')


# Compute the Forward index prices
Forwards = []
for i in range(0, 2):
    F = ATM_strikes[i] + np.exp(Rs[i]*Ts[i])*(ATM_calls[i]-ATM_puts[i])
    Forwards.append(F)

# Find K0, the strike price equal to or immediately below the forward index level F
K0s = []
for i in range(0, 2):
    K0 = np.maximum.reduce(Ks[Ks <= Forwards[i]])
    K0s.append(K0)



# Table of all calls above K0 and non-zeros, puts under K0 and non-zeros and the put-call average for K0 for near term options
K0_prices_near = Table_Near.loc[Table_Near['Strike'] == K0s[0]]
put_call_avg_near = float((K0_prices_near['Calls']+K0_prices_near['Puts'])/2)
calls_near = Table_Near[(Table_Near['Strike'] > K0s[0]) & (Table_Near['Calls'] != 0)][['Strike', 'Calls']]
puts_near = Table_Near[(Table_Near['Strike'] < K0s[0]) & (Table_Near['Puts'] != 0)][['Strike', 'Puts']]
options_near = calls_near.merge(puts_near, on='Strike', how='outer')
options_near['Option Price'] = options_near['Calls'].fillna(options_near['Puts'])
options_near = options_near.drop(['Calls', 'Puts'], axis=1)
new_row_near = pd.DataFrame({'Strike': [K0s[0]], 'Option Price': [put_call_avg_near]})
options_near = pd.concat([options_near, new_row_near], ignore_index=True)
options_near = options_near.sort_values('Strike')
options_near = options_near.reset_index(drop=True)




# Table of all calls above K0 and non-zeros, puts under K0 and non-zeros and the put-call average for K0 for next term options
K0_prices_next = Table_Next.loc[Table_Next['Strike'] == K0s[1]]
put_call_avg_next = float((K0_prices_next['Calls']+K0_prices_next['Puts'])/2)
calls_next = Table_Next[(Table_Next['Strike'] > K0s[1]) & (Table_Next['Calls'] != 1)][['Strike', 'Calls']]
puts_next = Table_Next[(Table_Next['Strike'] < K0s[1]) & (Table_Next['Puts'] != 1)][['Strike', 'Puts']]
options_next = calls_next.merge(puts_next, on='Strike', how='outer')
options_next['Option Price'] = options_next['Calls'].fillna(options_next['Puts'])
options_next = options_next.drop(['Calls', 'Puts'], axis=1)
new_row_next = pd.DataFrame({'Strike': [K0s[1]], 'Option Price': [put_call_avg_next]})
options_next = pd.concat([options_next, new_row_next], ignore_index=True)
options_next = options_next.sort_values('Strike')
options_next = options_next.reset_index(drop=True)


print('K0 are' , K0s)

# Computing the CBOE Volatility Index for the near-term options
# Compute Delta_K for the near-term options
delta_K_near = pd.Series(0, index=options_near.index)
delta_K_near.iloc[0] = options_near['Strike'].iloc[1]-options_near['Strike'].iloc[0]
delta_K_near.iloc[-1] = options_near['Strike'].iloc[-2]-options_near['Strike'].iloc[-1]
delta_K_near.iloc[1:-1] = (options_near['Strike'].iloc[2:].reset_index(drop=True) - options_near['Strike'].iloc[:-2].reset_index(drop=True)) / 2
delta_K_near = pd.DataFrame(delta_K_near.abs())


# Compute the square of Strike prices
K_square_near = pd.DataFrame(options_near['Strike']**2)
K_square_near.columns = [0]
e_rt_near = np.exp(R1*T1)  # Compute e^RT for near-term options
Contribution_by_Strike_near = delta_K_near[0]/K_square_near[0] * e_rt_near * options_near['Option Price']  # Contribution by strike for the near-term options

vix_near_1 = 2/T1 * Contribution_by_Strike_near.sum()      # First formula part of the options volatility calculation for the near-term options
vix_near_2 = 1/T1 * (Forwards[0]/K0s[0]-1)**2

vix_near_term = vix_near_1 - vix_near_2


# Computing the CBOE Volatility Index for the next-term options
# Compute Delta_K for the near-term options
delta_K_next = pd.Series(0, index=options_next.index)
delta_K_next.iloc[0] = options_next['Strike'].iloc[1]-options_next['Strike'].iloc[0]
delta_K_next.iloc[-1] = options_next['Strike'].iloc[-2]-options_next['Strike'].iloc[-1]
delta_K_next.iloc[1:-1] = (options_next['Strike'].iloc[2:].reset_index(drop=True) - options_next['Strike'].iloc[:-2].reset_index(drop=True)) / 2
delta_K_next = pd.DataFrame(delta_K_next.abs())


# Compute the square of Strike prices
K_square_next = pd.DataFrame(options_next['Strike']**2)
K_square_next.columns = [0]


e_rt_next = np.exp(R2*T2)  # Compute e^RT for next-term options
Contribution_by_Strike_next = delta_K_next[0]/K_square_next[0]  * e_rt_next * options_next['Option Price'] # Contribution by strike for the next-term options

vix_next_1 = 2/T2 * Contribution_by_Strike_next.sum()      # First formula part of the options volatility calculation for the next-term options
vix_next_2 = 1/T2 * (Forwards[1]/K0s[1]-1)**2

vix_next_term = vix_next_1 - vix_next_2


# Computing the number of minutes until expiration of the options
MT1 = (21*24*60 + (24-10)*60-5 + 9*60 + 30)          # Near-term options
MT2 =  (28*24*60 + (24-10)*60-5 + 16*60)             # Next-term options
MCM = 30*24*60                                       # Given constant maturity term (30 days)
M365 = 365*24*60                                     # Number of minutes in a year


# Computing the CBOE VIX Index
CBOE_VIX = 100 * np.sqrt(T1*vix_near_term * ((MT2-MCM)/(MT2-MT1)) + (T2*vix_next_term * ((MCM-MT1)/(MT2-MT1))) * M365/MCM)
print(CBOE_VIX)
print(f'The estimated CBOE Vix Index using simulated options prices with monte carlo simulations by using a GBM model is {CBOE_VIX:.2f}')


import numpy as np

# CMT data for available maturities
maturities = [30, 60, 91, 182, 365, 730, 1095, 1825, 2555, 3650, 7300, 10950]
yields = [4.74, 4.79, 4.85, 4.97, 4.94, 4.64, 4.06, 3.81, 3.60, 3.55, 3.48, 3.81]

# input
t = 22 # Input number of days


# Calculate slopes and intercepts for lower and upper bounds of BEY
def calc_bounds():
    m0_lower = (yields[1] - yields[0]) / (maturities[1] - maturities[0])
    b0_lower = yields[0] - m0_lower * maturities[0]
    m0_upper = (yields[-1] - yields[-2]) / (maturities[-1] - maturities[-2])
    b0_upper = yields[-1] - m0_upper * maturities[-1]
    return m0_lower, b0_lower, m0_upper, b0_upper

# Calculate BEY for a given maturity
def bey(t):
    # Check if the maturity is within available range
    if t in maturities:
        i = maturities.index(t)
        return ((1 + yields[i] * t / 36000) ** (360 / t) - 1) * 100
    else:
        # Extrapolate lower and upper bounds of BEY
        m0_lower, b0_lower, m0_upper, b0_upper = calc_bounds()
        if t < maturities[0]:
            bey_lower = m0_lower * t + b0_lower
            return ((1 + bey_lower * t / 36000) ** (360 / t) - 1) * 100
        elif t > maturities[-1]:
            bey_upper = m0_upper * t + b0_upper
            return ((1 + bey_upper * t / 36000) ** (360 / t) - 1) * 100
        else:
            # Extrapolation not possible for intermediate maturities
            return None

bey = bey(t)/100
print(bey)

# Converting Bey to a Continuously Compounded APY Rate
APY = (1 + bey/2)**2 - 1
r = np.log(1+APY)
print(r)
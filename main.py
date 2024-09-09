import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date
from scipy.optimize import minimize

today = date.today().strftime('%Y-%m-%d')

tickers = ['MSFT', 'AAPL', 'AMZN', 'NVDA', 'CSCO',
           'ORCL', 'AMD', 'QCOM', 'BLK', 'JPM', 'NFLX', 'TSLA']

def musd(x):
    m, n = x.shape
    mu = np.mean(x, axis=0)
    cv = np.cov(x.T)
    sd = np.sqrt(np.diag(cv))
    return sd, mu, cv

def MinVar(x):
    sd,mu,cvs = musd(x)
    cv = (2.0*cvs).tolist()
    n=len(cv)
    for i in range(n):
        cv[i].append(1.0)
    cv.append(np.ones(n).tolist() + [0])
    B = np.array(np.zeros(n).tolist() + [1.0])
    weights =  np.linalg.inv(cv).dot(B)[:-1]
    risk_x = np.sqrt(weights.T.dot(cvs.dot(weights)))
    retz_x = weights.T.dot(mu)
    return risk_x, retz_x

def MaxSharpe(o):
    def Optimize(cv,mu):
        def Objective(x):
            return -(x.T.dot(mu))/(x.T.dot(cv.dot(x)))
        def Constraint(x):
            return np.sum(x) - 1.0
        
        cons = [{'type': 'eq','fun': Constraint}]
        x = np.ones(len(mu))
        res = minimize(Objective,x,method='SLSQP',bounds=None,constraints=cons)
        return res.x
    
    sd,mu,cv = musd(o)
    weights = Optimize(cv,mu)
    risk_x = np.sqrt(weights.T.dot(cv.dot(weights)))
    retz_x = weights.T.dot(mu)

    return risk_x,retz_x

def EF(x):

    def Optimize(cov, mu, r):
        cov = (2.0*cov).tolist()
        n = len(cov)
        for i in range(n):
            cov[i].append(mu[i])
            cov[i].append(1.0)
        cov.append(mu.tolist() + [0,0])
        cov.append(np.ones(n).tolist()+[0,0])
        B = np.zeros(n).tolist() + [r,1.0]
        cov,B =np.array(cov), np.array(B)
        return np.linalg.inv(cov).dot(B)[:-2]
    
    sd, mu, cv = musd(x)
    ux, uy = [], []
    for i in np.arange(np.min(mu), np.max(mu) + 0.000001, 0.000001):
        weights = Optimize(cv, mu, i)
        ux.append(np.sqrt(weights.T.dot(cv).dot(weights)))
        uy.append(weights.T.dot(mu))

    return ux, uy

def CapitalAllocationLine(risk_x,risk_y):
    w = (0.5,1,1.5)
    x, y = [],[]
    for weight in w:
        x.append(weight*risk_x)
        y.append(weight*risk_y + (1-weight)*-risk_y)
    return x,y

# Download and save stock data
for ticker in tickers:
    data = yf.download(ticker, start="2023-01-01", end=today)
    data.to_csv(f"{ticker}.csv")

# Read the downloaded data into a dictionary of DataFrames
data = {tick: pd.read_csv(f'{tick}.csv') for tick in tickers}

# Extract adjusted close prices and calculate daily returns
close = np.array([data[tick]['Adj Close'].values for tick in tickers]).T

for tick in tickers:
    print(f"{tick} data: {data[tick].head()}")

# Calculate ror, mean, standard deviation and covariance of returns
ror = close[1:] / close[:-1] - 1.0

x, y, E = musd(ror)

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(x, y)

for i, ticker in enumerate(tickers):
    ax.annotate(ticker, (x[i], y[i]))

# Efficient frontier
ex, ey = EF(ror)
ax.plot(ex, ey, color='red', linewidth=0.8)

mx,my = MinVar(ror)
ax.scatter(mx,my,color='green', s=12)

sx,sy = MaxSharpe(ror)
ax.scatter(sx,sy,color='limegreen',s=12)

cx,cy = CapitalAllocationLine(sx,sy)
ax.plot(cx,cy,color='black',linewidth=0.8)

ax.set_xlabel('Standard Deviation (Risk)')
ax.set_ylabel('Mean Daily Return')
ax.set_title('Efficient Fronter Curve')

plt.show()


import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime
from time import time
import pytz
import scipy.optimize as sco
from sklearn.decomposition import PCA
from sklearn import cluster, covariance, ensemble, metrics, preprocessing
from collections import Counter
import WebScraperFinance as wsf # imported from WebScraperFinance.py file


# Python Environment: 3.5 interpreter
# first version: 01/03/2016
# last updated 01/11/2016
# Project: Quant Finance Modeling

## PROBLEM 1: Portfolio Examination and Optimization 


def portfolio_stats(portfolio,weights):   
    
    LogReturns = np.log(portfolio / portfolio.shift(1))
    AnnPerformance = LogReturns.mean() * 252 # annualized performance (252 trading days/year in US)
    CovPortfolio = LogReturns.cov() * 252
    weights = np.array(weights)
    ExpectPortfolioReturn = np.sum(LogReturns.mean() * weights) * 252   
    ExpectPortfolioSD = np.sqrt(np.dot(weights.T, np.dot(LogReturns.cov() * 252, weights))) # portfolio volatility=SD
    Pstats = np.array([ExpectPortfolioReturn, ExpectPortfolioSD, ExpectPortfolioReturn / ExpectPortfolioSD]) # element [2] is Sharp Ratio    
    return (Pstats,LogReturns,AnnPerformance,CovPortfolio)



def portfolio_montecarlo (n_sim, n_stocks,LogReturns):
    ExpectPortfolioReturn = []
    ExpectPortfolioSD = []
    for i in range (n_sim):
      weights = np.random.random(n_stocks); weights /= np.sum(weights)
      ExpectPortfolioReturn.append(np.sum(LogReturns.mean() * weights) * 252)
      ExpectPortfolioSD.append(np.sqrt(np.dot(weights.T, np.dot(LogReturns.cov() * 252, weights))))
    ExpectPortfolioReturn = np.array(ExpectPortfolioReturn)
    ExpectPortfolioSD = np.array(ExpectPortfolioSD)
    Sharp_Ratio=ExpectPortfolioReturn / ExpectPortfolioSD
    return(ExpectPortfolioReturn,ExpectPortfolioSD,Sharp_Ratio)

def stocks_moving_window(portfolio,w1,w2,marketref_stock):
    stock_ref = portfolio[marketref_stock]    
    stockstr=list(portfolio.columns.values)
    w1str='d%d' % w1; w2str='d%d' % w2; w1w2diff= w1str + '-' + w2str; 
    dic={}
    pch = portfolio.pct_change() 
    covariance_pch=pch.cov()
    variance_pch=pch[marketref_stock].var()
    for stock in portfolio:
      StockRoll = pd.DataFrame({'Close': portfolio[stock],w1str: pd.rolling_mean(portfolio[stock], w1), w2str: pd.rolling_mean(portfolio[stock], w2)})    
      StockRoll [w1w2diff] = StockRoll[w1str]-StockRoll[w2str] 
      SD = pd.Series.std(StockRoll[w2str])
      StockRoll['Regime'] = np.where(StockRoll[w1w2diff] > SD, 1, 0)
      StockRoll['Regime'] = np.where(StockRoll[w1w2diff] < -SD, -1, StockRoll['Regime'])
      # calculate BETA       
      beta = covariance_pch.loc[marketref_stock, stock] / variance_pch # will not be stored just plotted
      plt.clf()
      plt.figure(num=None, figsize=(20, 10), dpi=300, facecolor='w', edgecolor='k')
      plt.plot(StockRoll)  
      plt.legend(StockRoll.columns.values)
      plt.title(stock + ' (BETA: ' + str(beta.round(6)) +')', fontsize=24)       
      plt.savefig(stock+'_'+ w1w2diff +'.png', dpi=300)
      plt.clf()
      plt.close()
      StockRoll['LogReturn']  = np.log(StockRoll['Close'] / StockRoll['Close'].shift(1))
      StockRoll['Strategy']  = StockRoll['Regime'].shift(1)* StockRoll['LogReturn']      
      dic[stock] = StockRoll # collect results in dic  
    return(pd.Panel(dic))

def get_pca(portfolio):
    portfolio = portfolio.dropna()
    standardize_function = lambda x: (x - x.mean()) / x.std()
    portfolio = portfolio.apply(standardize_function)    
    pca = PCA().fit(portfolio)
    ExplVar=pca.explained_variance_ratio_      
    ScorePCA=PCA().fit_transform(portfolio)
    return (ExplVar,ScorePCA)

def process_s(s):
    s=s.dropna(axis=2, how='any')  # drop stocks that have NaNs, that is, stock data for specified range was not available from yahoo finance  
    for item in ['Open', 'High', 'Low']:
            s[item] = s[item] * s['Adj Close'] / s['Close']
    s.drop(['Close'], inplace=True)
    s.rename(items={'Adj Close': 'Close'}, inplace=True)    
    s['PercChange']=s['Close'].pct_change()
    s['MarketCap']=s['Close']*s['Volume']
    s['Range']=(s['High']-s['Low'])      
    s=s.fillna(method='backfill') 
    return (s)

def main():
    
    # define dates and stocks   
    start = datetime.datetime(2012, 5, 5, 0, 0, 0, 0, pytz.utc)    
    end = datetime.datetime.today().utcnow()
    
    # select from a large list, e.g. all S&P 500 stocks
    # scrape wikipedia S&P 500 site and get all tickers and a dict tickers per sector and all links about company
    site = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    pure_tickers,all_info_dict = wsf.scrape_get_tickers(site)
    # or define stocks manually 
    stocks = ['^GSPC','F','AAPL', 'IBM','YHOO'] 
    #stocks = pure_tickers
    #stocks.extend(['^GSPC']) # add reference market
    
    print('Downloading historical data from yahoo finance...')        
    # get daily stock info. Note this project does not include analysis of tickers
    s = web.DataReader(stocks, 'yahoo', start, end)   
    #s_actions = web.DataReader(stocks, 'yahoo-actions', start,end)
    
    s = process_s(s)    
    print('The following stocks will be analyzed: ', s.Close.columns)
    
    ## Clustering of all stocks (unsupervised) based on daily Range values (assume daily range/variation is most informative feature)
    # preprocess
    Dail_Volat = preprocessing.scale(np.array(s['Range']))  
    Dail_Volat = Dail_Volat.copy().T
    n_samples, n_features = Dail_Volat.shape # n_samples are stocks, n_features are daily ranges for n days    
    n_clusters=2 # makes more sense when downloading more stocks like all SP 500 stocks, then maybe 10
    pca = PCA(n_components=n_clusters).fit(Dail_Volat)

    # Kmeans clustering    
    clK = cluster.KMeans(init=pca.components_, n_clusters=n_clusters, n_init=1).fit(Dail_Volat)
    labels = clK.labels_    
    print ('Number of stocks per KMeans-Cluster: ', Counter([x for x in labels]))    
    ClusterDF=pd.DataFrame({'label':labels},index=s['Range'].columns.values) 
    #Cluster1Stocks=ClusterDF[ClusterDF.label == 1]
    print ('Clustering of stocks done')      
                       
    ## Optimization of given portfolio using Close values
    portfolio=s['Close']
   
    # Exploring portfolio via PCA
    ExplVar,ScorePCA=get_pca(portfolio)

    # Optimize weights of portfolio     
    n_stocks=len(portfolio.columns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # all weights have to sum up to 1
    bounds = [(0, 1) for x in range(n_stocks)] # restrict weights to be between 0 and 1
    Initialguess = n_stocks * [1. / n_stocks,] # initial guess to start simulation, choose all weights to be equal
    
    # the following optimizations currently only works in interactive mode and portfolio named portfolio!
    def min_func_sharpe(weights):
      return -portfolio_stats(portfolio,weights)[0][2] # max Sharp Ratio, that is, minimize neg.

    def min_func_variance(weights):
      return portfolio_stats(portfolio,weights)[0][1]**2 # **2 convert SD back to var


    Popt_W = sco.minimize(min_func_sharpe, Initialguess, method='SLSQP', bounds=bounds, constraints=constraints) 
    Popt_Var = sco.minimize(min_func_variance, Initialguess, method='SLSQP', bounds=bounds, constraints=constraints) 
    # see results
    print ('Optimum weights of portfolio: ', Popt_W['x'].round(4))
    portfolio.head()
    print ('Optimum variance of portfolio: ', Popt_Var['x'].round(4))
    
    
    
    # get basic stats of portfolio 
    Pstats,LogReturns,AnnPerformance,CovPortfolio=portfolio_stats(portfolio,Initialguess)
    
    print('MonteCarlo Simulation in process')
    # Monte Carlo Simulation >> generating random portfolio weight vectors on a larger scale and calc expected return and volatility
    n_sim=1000
    ExpectPortfolioReturn,ExpectPortfolioSD, Sharp_Ratio = portfolio_montecarlo (n_sim, n_stocks,LogReturns) # outputs are arrays length n_sim
    

    # Trad analysis of stock movement (moving averages, etc)
    w1=42; w2=252
    marketref = '^GSPC'
    print('Plots will be generated...')
    portfolio_mov=stocks_moving_window(s['Close'],w1,w2,marketref) # returns pandas panel with stocks as items
    print('Program done')
    # also saves plots (rolling moving windows tiem series and betas) for each stock in current directory!



if __name__ == '__main__':
    main()
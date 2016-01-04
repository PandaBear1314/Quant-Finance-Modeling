import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
import pytz
from sklearn import ensemble, metrics, preprocessing, cross_validation,feature_selection
import WebScraperFinance as wsf # imported from WebScraperFinance.py file

# Python Environment: 3.5 interpreter
# last updated 01/03/2016
# Project: Quant Finance Modeling

## PROBLEM 2: Forecasting Financial Time Series 


def main():
    
    # define dates and stocks   
    start = datetime.datetime(2015, 1, 4, 0, 0, 0, 0, pytz.utc)    
    end = datetime.datetime.today().utcnow()
    end = datetime.datetime(2015, 12, 2, 0, 0, 0, 0, pytz.utc)
    print('Daily stock prices will be downloaded from', start ,'to', end )

    # select from a large list, e.g. all S&P 500 stocks
    # scrape wikipedia S&P 500 site and get all tickers and a dict that contains tickers, sector, industry, and links to company website/reports
    site = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    pure_tickers,all_info_dict = wsf.scrape_get_tickers(site)
    # or define stocks manually 
    # stocks = ['^GSPC','F','AAPL', 'IBM','YHOO'] 
    stocks = pure_tickers
    stocks.extend(['^GSPC']) # add reference market
    
    print('Downloading historical data from yahoo finance...')     
    # get daily stock info (only historical daily data). 
    s = web.DataReader(stocks, 'yahoo', start, end)   
    #s_actions = web.DataReader(stocks, 'yahoo-actions', start,end)
   
    s=s.dropna(axis=2, how='any')  # drop stocks that have NaNs, that is, stock data for specified range was not available from yahoo finance  
    for item in ['Open', 'High', 'Low']:
            s[item] = s[item] * s['Adj Close'] / s['Close']
    s.drop(['Close'], inplace=True)
    s.rename(items={'Adj Close': 'Close'}, inplace=True)    
    s['PercChange']=s['Close'].pct_change()
    s['MarketCap']=s['Close']*s['Volume']
    s['Range']=(s['High']-s['Low'])      
    s=s.fillna(method='backfill')     
    print('The following SP 500 stocks will be analyzed: ', s.Close.columns)
    
    print('Deriving features...')
    # collect and prepare data: derive features / preprocessing
    shift=0 # backward shift from end of time series = day you want to predict/test model
    test_day=len(s.Close)-shift
    train_day_minus_two=test_day-2
    train_day_minus_one=test_day-1       
    
    # feature derivation and selection: I have no finance background, so this is going to be interesting...

    # feature 1: sector (categorical), extract from wikipedia web scraper
    sectorList=[]
    for stock in s.Volume.columns[:-1]:        
        for item in all_info_dict:
          if item['stock'] == stock:
              #print (stock, 'matches', item['stock'], 'and sector is:', item['sector'])
              sectorList.append(item['sector']) # 1d for encoding    
    le = preprocessing.LabelEncoder();sectorLabel=le.fit(sectorList); sectorListEnc=le.transform(sectorList) # le.inverse_transform(sectorListEnc) #list(le.classes_)
    sectorEnc=np.reshape(sectorListEnc,(len(sectorListEnc),1))
      
    # feature 2: BETA (using whole time series data before train_day)
    marketref='^GSPC'        
    pch = s.Close.iloc[:train_day_minus_two].pct_change()
    covariance_pch=pch.cov()
    variance_pch=pch[marketref].var()
    beta = np.array([[covariance_pch.loc[marketref, stock] / variance_pch] for stock in s.Volume.columns[:-1]]) # GSPC not used as stock  
    
    
    # features 3-8: Volume, LogReturns-1, LogReturns-2, Range-1, Range-2, EMA(Exponential Moving Average) only one iteration - close price
    LogReturns = np.log(s.Close / s.Close.shift(1))
    LogReturns=LogReturns.fillna(method='backfill') 
    Twindow=21;factor1=2/(Twindow+1);factor2=1-2/(Twindow+1)
    EMAstart_mean=s.Close.iloc[train_day_minus_two-3-Twindow:train_day_minus_two-3].mean()
   
    more_tech_indicat=np.array([[s['Volume'][stock][train_day_minus_two-1], LogReturns[stock][train_day_minus_two-1],LogReturns[stock][train_day_minus_two-2],\
                                    s['Close'][stock][train_day_minus_two-1]-s['Open'][stock][train_day_minus_two-1],\
                                    s['Close'][stock][train_day_minus_two-2]-s['Open'][stock][train_day_minus_two-2],\
                                    (s['Close'][stock][train_day_minus_two-2]*factor1+EMAstart_mean[stock]*factor2)-s['Close'][stock][train_day_minus_two-1]]\
                                    for stock in s.Volume.columns[:-1]])    
    

    # prepare target data
    binarizer = preprocessing.Binarizer().fit(LogReturns)
    a= binarizer.transform(LogReturns) 
    target1a=a[train_day_minus_two,:-1] # excluded GSPC stock at the end
    target2=a[train_day_minus_one,:-1] # excluded GSPC stock at the end
    target1=np.array(target1a).reshape(beta.shape)
    target2=np.array(target2).reshape(beta.shape)
    
    # first combine all features and targets in one array and shuffle, then split again

    data = np.concatenate((target1,target2,sectorEnc,beta,more_tech_indicat), axis=1)
    np.random.shuffle(data)
    target1=np.array(data[:,[0]]).reshape(target1a.shape).astype(np.int,copy=False) # make sure categorical is int  
    target2=np.array(data[:,[1]]).reshape(target1a.shape).astype(np.int,copy=False)
    data_temp = data[:,3:] 
    
    # feature selection, remove features with low variance
    sel = feature_selection.VarianceThreshold(threshold=(.8 * (1 - .8)))
    data_temp = sel.fit_transform(data_temp) # some of the original features don't appear to be useful
    data = np.concatenate((sectorEnc,data_temp), axis=1).astype(np.object, copy=False)      
    data[:,0]=data[:,0].astype(np.int,copy=False) # make sure categorical is int
     
       
    # partitioning data sets
    Xtrain, Xval, Y1train, Y1val, Y2train, Y2val = cross_validation.train_test_split(data, target1,target2, train_size=0.6)
    # right now I don't use Y2train, Y2val
  
    ## Machine Learning Classification
    print('Machine Learning Classification...predicting direction of LogReturn movement of day:', end-datetime.timedelta(days=2) )
    # Random Forest predict train_day_minus_two
    clf_RF = ensemble.RandomForestClassifier(n_estimators=1000,n_jobs=2)
    clf_RFfit = clf_RF.fit(Xtrain, Y1train)
    y_predRF_InSample = clf_RFfit.predict(Xtrain)
    y_predRF_OutSample = clf_RFfit.predict(Xval)

    print ('RF confusion matrix In Sample: ', '\n', metrics.confusion_matrix(Y1train,  y_predRF_InSample))
    print ('RF accuracy In Sample: ',metrics.accuracy_score(Y1train,  y_predRF_InSample, normalize=True, sample_weight=None))
    print ('RF confusion matrix Out Sample/Validation: ', '\n', metrics.confusion_matrix(Y1val,  y_predRF_OutSample))
    print ('RF accuracy Out Sample/Validation: ', metrics.accuracy_score(Y1val,  y_predRF_OutSample, normalize=True, sample_weight=None))

    ## To do: A lot, read up more on finance to derive better features, include much more stocks from different markets, etc; 
    ## extend scraping the web for useful info beyond yahoo finance, blend various classifiers, try multioutput, etc

    
    

if __name__ == '__main__':
    main()
# StockPrediction
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Microsoft Excel](https://img.shields.io/badge/Microsoft_Excel-217346?style=for-the-badge&logo=microsoft-excel&logoColor=white)  
#### Predicting the stock market using python and machine learning
+ Import yfinance American data
```python
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
sp500=yf.Ticker("^GSPC")
spf500=sp500.history(period="max")
spf500.index
```
![pycharmdata1](https://github.com/lynnhr/TwitterScrape/assets/151964289/1a9e4f49-067a-4686-b4e9-17cbb1bf57e0)
![pycharmChart1](https://github.com/lynnhr/TwitterScrape/assets/151964289/7b5080c7-41fb-4d29-a929-75f02202bdf5)

+ Delete unnecessary rows
```python
del spf500["Dividends"]
del spf500["Stock Splits"]
```
### Setting Up Our Target For Machine Learning  
+ Shift the cells and Create a Target Column
```python
spf500["Tomorrow"]=spf500["Close"].shift(-1)
spf500["Target"] = (spf500["Tomorrow"]>spf500["Close"]).astype(int)
spf500=spf500.loc["1990-01-01":].copy()
```
![pycharmData2](https://github.com/lynnhr/TwitterScrape/assets/151964289/749d8145-d6b1-459d-b435-b95c9beaa079)

### Training An Initial Machine Learning Model  
+ Use Random Forest Tree and Split data into train and test set
```python
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)
train=spf500.iloc[:-300] #put all the rows except the last 300 into the training set
test=spf500.iloc[-300:]  #put the las 300 into the test set
```
+ Train the model to predict
```python
predictors=["Close","Volume","Open","High","Low"]
model.fit(train[predictors],train["Target"])
from sklearn.metrics import precision_score
preds=model.predict(test[predictors])
```
+ Measure the performance
```python
preds=pd.Series(preds,index=test.index)
precision=precision_score(test["Target"],preds)
combined=pd.concat([test["Target"],preds],axis=1) #axis=1 treat them as columns
print(precision)
combined.plot()
plt.show()
```
0.5528455284552846  
![pycharmChart2](https://github.com/lynnhr/TwitterScrape/assets/151964289/f785c4bc-b641-433c-9d0a-e05f178af617)

#### Backtesting
+ Wrap code into functions
```python
def predict(train,test,predictors,model):
    model.fit(train[predictors],train["Target"]) #fitting the model
    preds=model.predict(test[predictors]) #generating our predictions
    preds=pd.Series(preds,index=test.index,name="Predictions") #combining them into a series
    combined=pd.concat([test["Target"],preds],axis=1)
    return combined
#step predict a year after a year(take the first 10 years of data then predict the 11th, then take the 11 years of data then predict the 12th...)
def backtest(data,model,predictors,start=2500,step=250):
    all_predictions=[]
    for i in range(start,data.shape[0],step): #the loop starts with i=2500(size of the training set (10 years) data.shape[0] gives the number of rows in a dataframe
        train= data.iloc[0:i].copy() #all the years except the current year
        test=data.iloc[i:(i+step)].copy()
        predictions=predict(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

predictions=backtest(spf500,model,predictors)
p1=predictions["Predictions"].value_counts() #count how many times each prediction was made (0 and 1)
p2=precision_score(predictions["Target"],predictions["Predictions"])
p3=predictions["Target"].value_counts() / predictions.shape[0]
```
Predictions  
0     :   3581  
1    :    2596    
0.5288906009244992  
Target  
1     :    0.534887  
0      :   0.465113    
![pycharmData3](https://github.com/lynnhr/TwitterScrape/assets/151964289/6b986039-5ca6-4f89-8534-5ca20de13105)


#### Adding Additional Predictors To Our Model
```python
#creating variety of rolling averages, calculate the mean close price in the last2 days...,then find the ratio to know market upturn od upswing
horizons=[2,5,60,250,1000]
new_predictors=[]
for horizon in horizons:
    rolling_averages=spf500.rolling(horizon).mean()
    ratio_column=f"Close_Ratio_{horizon}"
    spf500[ratio_column]=spf500["Close"] / rolling_averages["Close"] #the ratio between today's close and average close in the last 2 days for first horizon
    #trend: number of days the close price actually went up
    trend_column=f"Trend_{horizon}"
    spf500[trend_column]=spf500.shift(1).rolling(horizon).sum()["Target"] #this gonna find the sum of the targets of the last few days of a chosen certain day

    new_predictors+=[ratio_column,trend_column]
#get rid of extra missing rows NaN
spf500=spf500.dropna()
```

### Improving Our Model
```python
model=RandomForestClassifier(n_estimators=200,min_samples_split=50,random_state=1)
def predict(train,test,predictors,model):
    model.fit(train[predictors],train["Target"]) #fitting the model
    preds=model.predict_proba(test[predictors]) [:,1]#generating our predictions using predic_proba wich will return the probability that the market will be zero or 1
    preds[preds >= .6]=1 #60% confident
    preds[preds < .6]=0
    preds=pd.Series(preds,index=test.index,name="Predictions") #combining them into a series
    combined=pd.concat([test["Target"],preds],axis=1)
    return combined

predictions=backtest(spf500,model,new_predictors)
p4=predictions["Predictions"].value_counts()

p5=precision_score(predictions["Target"],predictions["Predictions"])
```
57%


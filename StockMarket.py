import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

sp500=yf.Ticker("^GSPC")
spf500=sp500.history(period="max")
spf500.index

del spf500["Dividends"]
del spf500["Stock Splits"]

#---------------------Setting Up Our Target For Machine Learning---------------------

spf500["Tomorrow"]=spf500["Close"].shift(-1)
spf500["Target"] = (spf500["Tomorrow"]>spf500["Close"]).astype(int)
spf500=spf500.loc["1990-01-01":].copy()

#---------------------Training An Initial Machine Learning Model---------------------
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)
train=spf500.iloc[:-300] #put all the rows except the last 300 into the training set
test=spf500.iloc[-300:]  #put the las 300 into the test set
predictors=["Close","Volume","Open","High","Low"]
model.fit(train[predictors],train["Target"])
from sklearn.metrics import precision_score
preds=model.predict(test[predictors]) #this is gonna generate predictions
preds=pd.Series(preds,index=test.index)
precision=precision_score(test["Target"],preds)
#---------------------Backtesting--------------------
def predict(train,test,predictors,model):
    model.fit(train[predictors],train["Target"]) #fitting the model
    preds=model.predict(test[predictors]) #generating our predictions
    preds=pd.Series(preds,index=test.index,name="Predictions") #combining them into a series
    combined=pd.concat([test["Target"],preds],axis=1)
    return combined

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

#!!!!!PRINT
# p1.head()
# print(p1)
# predictions.head()
# print(predictions)
# print(p2)
# p3.head()
# print(p3)
# spf500.head()
# print(spf500)

#---------------------Adding Additional Predictors To Our Model--------------------
horizons=[2,5,60,250,1000]
new_predictors=[]
for horizon in horizons:
    rolling_averages=spf500.rolling(horizon).mean()
    ratio_column=f"Close_Ratio_{horizon}"
    spf500[ratio_column]=spf500["Close"] / rolling_averages["Close"] #the ratio between today's close and average close in the last 2 days for first horizon

    trend_column=f"Trend_{horizon}"
    spf500[trend_column]=spf500.shift(1).rolling(horizon).sum()["Target"] #this gonna find the sum of the targets of the last few days of a chosen certain day

    new_predictors+=[ratio_column,trend_column]

spf500=spf500.dropna()


#-------------------Improving Our Model----------------------
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

#!!!!!PRINT
# print(p4)
# print(p5)
# spf500.to_csv("spf500.csv")

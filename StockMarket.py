#download daily stock index prices
import yfinance as yf

#for visualisation
import matplotlib.pyplot as plt
import pandas as pd

#initialise ticker class to enable us to download price history by symbol
sp500=yf.Ticker("^GSPC")
#quiery the historical prices
spf500=sp500.history(period="max")



#index to slice the dataframe easily
spf500.index

#!!!!!PRINT
# spf500.plot.line(y="Close",use_index=True)
# plt.show()


#cleaning up data
del spf500["Dividends"]
del spf500["Stock Splits"]

#---------------------Setting Up Our Target For Machine Learning---------------------

spf500["Tomorrow"]=spf500["Close"].shift(-1)
#we shifted the cell as you can see the tomorrow of 12/30 is the same as close of 1/3
#!!!!!PRINT
spf500.head()
print(spf500)

#based on tomorrow's price we are gonna setup a target
#is tomorrow's price greater than today's price
#return boolean
#astype method to convert it to integer
spf500["Target"] = (spf500["Tomorrow"]>spf500["Close"]).astype(int)


#remove all data before 1990 the market shifts irrelevant
#pandas loc only take the rows where index...
spf500=spf500.loc["1990-01-01":].copy()

#!!!!!PRINT
# print(spf500)
# spf500.head()

#---------------------Training An Initial Machine Learning Model---------------------
from sklearn.ensemble import RandomForestClassifier
#initisalize the model
#research about decision tree
#research cross-validation
#random_state=1 if we rerun the model twice we'll get the same result
model=RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)

#split data into train and test
train=spf500.iloc[:-300] #put all the rows except the last 300 into the training set
test=spf500.iloc[-300:]  #put the las 300 into the test set
predictors=["Close","Volume","Open","High","Low"]
#using the predictors columns then try to predict the target
model.fit(train[predictors],train["Target"])

#measure the performance
from sklearn.metrics import precision_score
preds=model.predict(test[predictors]) #this is gonna generate predictions

#these preds are in an array so we turn them into a pandas series
preds=pd.Series(preds,index=test.index)

#calculate the precision score using the actual target and the predicted target
precision=precision_score(test["Target"],preds)

#!!!!!PRINT
# combined=pd.concat([test["Target"],preds],axis=1) #axis=1 treat them as columns
# print(precision)
# combined.plot()
# plt.show()

#---------------------Backtesting--------------------
#wrap everything up into a funtion
def predict(train,test,predictors,model):
    model.fit(train[predictors],train["Target"]) #fitting the model
    preds=model.predict(test[predictors]) #generating our predictions
    preds=pd.Series(preds,index=test.index,name="Predictions") #combining them into a series
    combined=pd.concat([test["Target"],preds],axis=1)
    return combined

#write a backtest function
#start data to train the first model 2500 days(10 years of data)
#step predict a year after a year(take the first 10 years of data then predict the 11th, then take the 11 years of data then predict the 12th...)
def backtest(data,model,predictors,start=2500,step=250):
    all_predictions=[]
    for i in range(start,data.shape[0],step): #the loop starts with i=2500(size of the training set (10 years) data.shape[0] gives the number of rows in a dataframe
        train= data.iloc[0:i].copy() #all the years except the current year
        test=data.iloc[i:(i+step)].copy()
        predictions=predict(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Assume data has 5000 rows.
# First Iteration:
#
# i = 2500
# train = data.iloc[0:2500].copy(): This includes rows from 0 to 2499.
# test = data.iloc[2500:2750].copy(): This includes rows from 2500 to 2749.
# Second Iteration:
#
# i = 2750
# train = data.iloc[0:2750].copy(): This includes rows from 0 to 2749.
# test = data.iloc[2750:3000].copy(): This includes rows from 2750 to 2999.

predictions=backtest(spf500,model,predictors)
#evaluating our predictions
p1=predictions["Predictions"].value_counts() #count how many times each prediction was made (0 and 1)
p2=precision_score(predictions["Target"],predictions["Predictions"])
#look at the percentage of days where the market actually went up
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
#creating variety of rolling averages, calculate the mean close price in the last2 days...,then find the ratio to know market upturn od upswing
horizons=[2,5,60,250,1000]
new_predictors=[]
#loop through the horizon and calculate the average
for horizon in horizons:
    rolling_averages=spf500.rolling(horizon).mean()
    #create multiple columns
    ratio_column=f"Close_Ratio_{horizon}"
    spf500[ratio_column]=spf500["Close"] / rolling_averages["Close"] #the ratio between today's close and average close in the last 2 days for first horizon

    #trend: number of days the close price actually went up
    trend_column=f"Trend_{horizon}"
    #we are shifting forward this time
    spf500[trend_column]=spf500.shift(1).rolling(horizon).sum()["Target"] #this gonna find the sum of the targets of the last few days of a chosen certain day

    new_predictors+=[ratio_column,trend_column]

#get rid of extra missing rows NaN
spf500=spf500.dropna()

#!!!!!PRINT
# spf500.head()
# print(spf500)



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
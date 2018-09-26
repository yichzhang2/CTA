import urllib.request
import pandas as pd
import talib
import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm, model_selection
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
#from sklearn.svm import SVR
import numpy as np
from sklearn.externals import joblib
import datetime

def SVM_model(symbol,randomstate,predict_window=1):
    url = r'http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesMiniKLine5m?symbol='+symbol
    res = urllib.request.urlopen(url)
    html = res.read().decode('utf-8')

    temp=pd.read_json(html)
    temp.columns=['time','open','high','low','close','volumn']
    temp.sort_values(by='time',inplace= True)
    temp=temp.reset_index(drop=True)
    #sns.distplot(temp['volumn'], kde=False)

    #sns.plt.show()


    #ADX=talib.ADX(temp['high'], temp['low'], temp['close'], timeperiod=14)
    #plt.plot(ADX)

    temp['CDL2CROWS']=talib.CDL2CROWS(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDL3BLACKCROWS']=talib.CDL3BLACKCROWS(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDL3INSIDE']=talib.CDL3INSIDE(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDL3LINESTRIKE']=talib.CDL3LINESTRIKE(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDL3OUTSIDE']=talib.CDL3OUTSIDE(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDL3STARSINSOUTH']=talib.CDL3STARSINSOUTH(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDL3WHITESOLDIERS']=talib.CDL3WHITESOLDIERS(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLABANDONEDBABY']=talib.CDLABANDONEDBABY(temp['open'], temp['high'], temp['low'], temp['close'], penetration=0)
    temp['CDLADVANCEBLOCK']=talib.CDLADVANCEBLOCK(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLBELTHOLD']=talib.CDLBELTHOLD(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLBREAKAWAY']=talib.CDLBREAKAWAY(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLCLOSINGMARUBOZU']=talib.CDLCLOSINGMARUBOZU(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLCONCEALBABYSWALL']=talib.CDLCONCEALBABYSWALL(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLCOUNTERATTACK']=talib.CDLCOUNTERATTACK(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLDARKCLOUDCOVER']=talib.CDLDARKCLOUDCOVER(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLDOJI']=talib.CDLDOJI(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLDOJISTAR']=talib.CDLDOJISTAR(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLDRAGONFLYDOJI']=talib.CDLDRAGONFLYDOJI(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLENGULFING']=talib.CDLENGULFING(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLEVENINGDOJISTAR']=talib.CDLEVENINGDOJISTAR(temp['open'], temp['high'], temp['low'], temp['close'],penetration=0)
    temp['CDLEVENINGSTAR']=talib.CDLEVENINGSTAR(temp['open'], temp['high'], temp['low'], temp['close'], penetration=0)
    temp['CDLGAPSIDESIDEWHITE']=talib.CDLGAPSIDESIDEWHITE(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLGRAVESTONEDOJI']=talib.CDLGRAVESTONEDOJI(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLHAMMER']=talib.CDLHAMMER(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLHANGINGMAN']=talib.CDLHANGINGMAN(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLHARAMI']=talib.CDLHARAMI(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLHARAMICROSS']=talib.CDLHARAMICROSS(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLHIGHWAVE']=talib.CDLHIGHWAVE(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLHIKKAKE']=talib.CDLHIKKAKE(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLHIKKAKEMOD']=talib.CDLHIKKAKEMOD(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLHOMINGPIGEON']=talib.CDLHOMINGPIGEON(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLIDENTICAL3CROWS']=talib.CDLIDENTICAL3CROWS(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLINNECK']=talib.CDLINNECK(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLINVERTEDHAMMER']=talib.CDLINVERTEDHAMMER(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLKICKING']=talib.CDLKICKING(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLKICKINGBYLENGTH']=talib.CDLKICKINGBYLENGTH(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLLADDERBOTTOM']=talib.CDLLADDERBOTTOM(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLLONGLEGGEDDOJI']=talib.CDLLONGLEGGEDDOJI(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLLONGLINE']=talib.CDLLONGLINE(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLMARUBOZU']=talib.CDLMARUBOZU(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLMATCHINGLOW']=talib.CDLMATCHINGLOW(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLMATHOLD']=talib.CDLMATHOLD(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLMORNINGDOJISTAR']=talib.CDLMORNINGDOJISTAR(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLMORNINGSTAR']=talib.CDLMORNINGSTAR(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLONNECK']=talib.CDLONNECK(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLPIERCING']=talib.CDLPIERCING(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLRICKSHAWMAN']=talib.CDLRICKSHAWMAN(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLRISEFALL3METHODS']=talib.CDLRISEFALL3METHODS(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLSEPARATINGLINES']=talib.CDLSEPARATINGLINES(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLSHOOTINGSTAR']=talib.CDLSHOOTINGSTAR(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLSHORTLINE']=talib.CDLSHORTLINE(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLSPINNINGTOP']=talib.CDLSPINNINGTOP(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLSTALLEDPATTERN']=talib.CDLSTALLEDPATTERN(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLSTICKSANDWICH']=talib.CDLSTICKSANDWICH(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLTAKURI']=talib.CDLTAKURI(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLTASUKIGAP']=talib.CDLTASUKIGAP(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLTHRUSTING']=talib.CDLTHRUSTING(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLTRISTAR']=talib.CDLTRISTAR(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLUNIQUE3RIVER']=talib.CDLUNIQUE3RIVER(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLUPSIDEGAP2CROWS']=talib.CDLUPSIDEGAP2CROWS(temp['open'], temp['high'], temp['low'], temp['close'])
    temp['CDLXSIDEGAP3METHODS']=talib.CDLXSIDEGAP3METHODS(temp['open'], temp['high'], temp['low'], temp['close'])


    temp['MA5C'] = np.divide(talib.MA(temp['close'], timeperiod=5),temp['close'])
    temp['MA10C'] = np.divide(talib.MA(temp['close'], timeperiod=10),temp['close'])
    temp['MA20C'] = np.divide(talib.MA(temp['close'], timeperiod=20),temp['close'])
    temp['MA5V'] = np.divide(talib.MA(temp['volumn'], timeperiod=5),temp['volumn'])
    temp['MA10V'] = np.divide(talib.MA(temp['volumn'], timeperiod=10),temp['volumn'])
    temp['MA20V'] = np.divide(talib.MA(temp['volumn'], timeperiod=20),temp['volumn'])
    temp['RSI6'] = talib.RSI(temp['close'], timeperiod=6)
    temp['RSI12'] = talib.RSI(temp['close'], timeperiod=12)
    temp['RSI24'] = talib.RSI(temp['close'], timeperiod=24)
    temp['MOM5'] = talib.MOM(temp['close'], timeperiod=5)
    temp['MOM10'] = talib.MOM(temp['close'], timeperiod=10)

    temp_change = temp['close'].pct_change(periods=predict_window)
    temp_change=temp_change.shift(-1*predict_window)
    temp['pct_chg_shift']=temp_change

    max_abs_scaler = preprocessing.MaxAbsScaler()
    x_maxsbs = pd.DataFrame(max_abs_scaler.fit_transform(temp.iloc[:,6:67]))
    x_maxsbs.columns=temp.columns[6:67]
    x_maxsbs['time']=temp['time']
    x_maxsbs['pct_chg_shift']=(temp['pct_chg_shift']>0)*1+(temp['pct_chg_shift']<0)*(-1)

    temp_unscaled=temp.iloc[:,67:78]
    temp_unscaled['time'] = temp['time']
    temp_unscaled.dropna(axis=0,inplace=True)
    temp_unscaled=temp_unscaled.reset_index(drop=True)
    x_scaled = preprocessing.scale(temp_unscaled.iloc[:,0:11])
    x_scaled = pd.DataFrame(x_scaled)
    x_scaled['time']=temp_unscaled['time']
    x_scaled.columns=temp_unscaled.columns

    x_maxsbs=x_maxsbs.merge(x_scaled,on='time',how='left')

    x_maxsbs.dropna(axis=0,inplace=True)

    label=x_maxsbs['pct_chg_shift']

    x_maxsbs.drop(['time'],axis=1,inplace=True)

    x_maxsbs.drop(['pct_chg_shift'],axis=1,inplace=True)

    Xtrain, Xtest, Ytrain, Ytest =train_test_split(x_maxsbs, label,test_size=0.01, random_state=4)

    clf = svm.SVC(gamma=0.01, C=100.)
    clf.fit(Xtrain, Ytrain)
    Ypred = clf.predict(Xtest)

    Ypred=pd.DataFrame(Ypred)
    Ytest=pd.DataFrame(Ytest)
    #IC=Ytest['pct_chg_shift'].corr(Ypred[0],method='spearman')

    today=datetime.date.today().strftime("%Y%m%d")
    joblib.dump(clf, symbol+"_"+today+"_model.m")
    return accuracy_score(Ytest, Ypred)


#    clf = joblib.load("train_model.m")

SVM_model('rb0',randomstate=2)
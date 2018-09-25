import  urllib.request
import pandas as pd
import talib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm, model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#from sklearn.svm import SVR
import numpy as np


url = r'http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesDailyKLine?symbol=M0'
res = urllib.request.urlopen(url)
html = res.read().decode('utf-8')

temp=pd.read_json(html)
temp.columns=['time','open','high','low','close','volumn']
sns.distplot(temp['volumn'], kde=False)

sns.plt.show()


ADX=talib.ADX(temp['high'], temp['low'], temp['close'], timeperiod=14)
plt.plot(ADX)

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

predict_window=5
temp_change = temp['close'].pct_change(periods=predict_window)
temp_change=temp_change.shift(-1*predict_window)
temp['pct_chg_shift']=temp_change

max_abs_scaler = preprocessing.MaxAbsScaler()
x_maxsbs = pd.DataFrame(max_abs_scaler.fit_transform(temp.iloc[:,6:67]))
x_maxsbs.columns=temp.columns[6:67]
x_maxsbs['time']=temp['time']
x_maxsbs['pct_chg_shift']=(temp['pct_chg_shift']>0)*1+(temp['pct_chg_shift']<0)*(-1)

x_maxsbs.drop(np.where(np.isnan(x_maxsbs['pct_chg_shift']))[0],inplace=True)
x_maxsbs=x_maxsbs.iloc[np.nonzero(x_maxsbs.iloc[:,0:61].sum(axis=1))[0],:]

Xtrain, Xtest, Ytrain, Ytest =train_test_split(x_maxsbs.iloc[:,0:61], x_maxsbs.iloc[:,62],test_size=0.20, random_state=8)

clf = svm.SVC(gamma=0.01, C=100.)
clf.fit(Xtrain, Ytrain)
Ypred = clf.predict(Xtest)

Ypred=pd.DataFrame(Ypred)
Ytest=pd.DataFrame(Ytest)
#IC=Ytest['pct_chg_shift'].corr(Ypred[0],method='spearman')

accuracy_score(Ytest, Ypred)


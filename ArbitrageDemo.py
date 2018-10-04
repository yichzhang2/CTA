import urllib.request
import pandas as pd
import numpy as np
import datetime


def fetch_data(symbol,interval='Daily'):
    if interval=='Daily':
        url = r'http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesDailyKLine?symbol='+symbol
    elif interval=='5m':
        url = r'http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesMiniKLine5m?symbol=' + symbol
    elif interval=='30m':
        url = r'http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesMiniKLine30m?symbol=' + symbol
    elif interval=='60m':
        url = r'http://stock2.finance.sina.com.cn/futures/api/json.php/IndexService.getInnerFuturesMiniKLine30m?symbol=' + symbol
    res = urllib.request.urlopen(url)
    html = res.read().decode('utf-8')

    temp=pd.read_json(html)
    temp.columns=['time','open','high','low','close','volumn']
    temp.sort_values(by='time',inplace= True)
    temp=temp.reset_index(drop=True)
    return temp

def difference(symbol1,symbol2,threshold):
    temp1=fetch_data(symbol1)
    temp2=fetch_data(symbol2)

    temp1.time=pd.to_datetime(temp1.time)
    temp2.time = pd.to_datetime(temp2.time)

    threshold=datetime.datetime.strptime(threshold,"%Y-%m-%d")

    del_temp1=pd.DataFrame(temp1.time<threshold)
    del_temp1 = del_temp1[del_temp1['time'] == True].index
    temp1.drop(del_temp1,inplace=True)
    temp1 = temp1.reset_index(drop=True)

    del_temp2=pd.DataFrame(temp2.time<threshold)
    del_temp2 = del_temp2[del_temp2['time'] == True].index
    temp2.drop(del_temp2,inplace=True)
    temp2 = temp2.reset_index(drop=True)

    result=pd.DataFrame(temp1['close']-temp2['close'])
    result['time']=temp1['time']

    latesest_diff=result.close[len(result)-1]
    sample=len(result)
    min_time= result.time[0]
    max_time= result.time[len(result)-1]

    avg_diff= np.mean(result.close)
    std_diff= np.std(result.close)
    min_diff=np.min(result.close)
    avg_plus_2std=avg_diff+2*std_diff
    avg_minus_2std=avg_diff-2*std_diff
    max_diff=np.max(result.close)


    return latesest_diff, sample, min_time, max_time, avg_diff, std_diff, min_diff,avg_minus_2std,avg_plus_2std,max_diff




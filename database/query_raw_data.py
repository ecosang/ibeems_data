
#import yaml
#import re
import json

import pandas as pd
import numpy as np
import requests
import time
import os
import pathlib
import sys


repo_path=pathlib.Path(os.path.abspath(os.path.dirname(__file__)).__str__()).parents[0].absolute().__str__()
sys.path.insert(1,pathlib.Path(repo_path).absolute().__str__()) # repository path
#os.chdir(repo_path)

from database.function_ts import get_strtimestamp_dict

# get_strtimestamp_dict(tz,timestamp_utc=None,timestamp_local=None)

def raw_data_query(path,start_time_utc,end_time_utc,credential,init_config,query_split_days=1):

    '''
    start_time_utc: str
    '''
    tz=init_config['tz']
    Ts_raw=init_config['Ts_raw']
    # all time unit is in TZ time

    start_time=pd.Timestamp(start_time_utc)
    end_time=pd.Timestamp(end_time_utc)

    if start_time.tz is None:
        start_time=start_time.tz_localize("UTC").tz_convert(tz).strftime("%Y-%m-%dT%H:%M:%S")
    else:
        start_time=start_time.tz_convert(tz).strftime("%Y-%m-%dT%H:%M:%S")

    if end_time.tz is None:
        end_time=end_time.tz_localize("UTC").tz_convert(tz).strftime("%Y-%m-%dT%H:%M:%S")
    else:
        end_time=end_time.tz_convert(tz).strftime("%Y-%m-%dT%H:%M:%S")

    n_days=max(query_split_days,1)
    query_start_time=(pd.Timestamp(start_time)-pd.Timedelta("1 h")).strftime("%Y-%m-%dT%H:%M:%S")
    query_end_time=(pd.Timestamp(end_time)+pd.Timedelta("1 h")).strftime("%Y-%m-%dT%H:%M:%S")
    
    nxdays=np.floor((pd.Timestamp(query_end_time)-pd.Timestamp(query_start_time)).total_seconds()/(3600*24*n_days))
    ndays=np.floor((pd.Timestamp(query_end_time)-pd.Timestamp(query_start_time)).total_seconds()/(3600*24))
    
    if ndays==0:
        start_time_list=[query_start_time]
        end_time_list=[query_end_time]
    else:
        start_time_list=[query_start_time]
        end_time_list=[]
        if nxdays>0:
            for i in np.arange(nxdays):
                end_time_list.append((pd.Timestamp(query_start_time)+pd.Timedelta(f'{n_days}D')*(i+1)).strftime("%Y-%m-%dT%H:%M:%S"))        
                start_time_list.append((pd.Timestamp(query_start_time)+pd.Timedelta(f'{n_days}D')*(i+1)).strftime("%Y-%m-%dT%H:%M:%S"))
                
        end_time_list.append(query_end_time)
    
    df_list=[]
    for st,et in zip(start_time_list,end_time_list):
        endpoint_path=credential['endpoint'].format(path=path,fromDateTime=st,toDateTime=et)
        # here raw json or None is returned .
        data_list=query_endpoint(url=endpoint_path,credential=credential,n_try=2,sleep_time=0.1)
        if data_list is not None:
            df=pd.DataFrame(data_list)
            df.rename(columns={'timeStamp':'timestamp'},inplace=True)
            df_list.append(df)
    if len(df_list)!=0:
        df_rawData=pd.concat(df_list)
        df_rawData=df_rawData.reset_index(level=0,drop=True)
    else:
        df_rawData=None
    
    if df_rawData is not None:
        df_rawData['timestamp']=pd.to_datetime(df_rawData['timestamp']).dt.tz_localize(tz)
        df_rawData.rename(columns={'timestamp':'timestamp_local'},inplace=True)
        df_rawData['timestamp_local']=df_rawData['timestamp_local'].dt.round(pd.Timedelta(f'{int(Ts_raw)}sec'))
        df_rawData=df_rawData.drop_duplicates(subset=['timestamp_local'])

        #df_rawData['timestamp_local']=df_rawData['timestamp_local'].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        df_rawData=df_rawData.sort_values('timestamp_local').reset_index(drop=True)
        df_rawData.rename(columns={'value':path},inplace=True)
        
        
    

    return df_rawData





def query_endpoint(url,credential,n_try=2,sleep_time=0.2):
    try_error=True
    n_try_=1
    
    while try_error:
        try:
            # try query
            payload = {}
            headers = {
                'serviceToken': credential['serviceToken']#'XX514lgodZ'
            }
            response = requests.get(url, headers=headers, data=payload)
            if response.status_code == 200:
                textData=response.text
                if textData is None or textData=="":
                    #df_data=None
                    raise ValueError(f"No data received for {url}. data is empty {textData}")
                    
                else:
                    
                    data_list = json.loads(textData)
                    if not data_list:
                        raise ValueError(f"No data received for {url}")
                    else:
                        try_error=False #success
                        
            else:
                raise ValueError(f"{url}: Request failed with status code {response.status_code}")
        
        except Exception as e:
            time.sleep(sleep_time+n_try_/10)
            n_try_=n_try_+1
            try_error=True
            print(e)
            if n_try_>=n_try:
                print(f"after {n_try_}, it still failed. the error message is {e}. but, function continues..")
                try_error=False
                data_list=None
    return data_list

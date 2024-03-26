

import pandas as pd
import yaml
import json
import numpy as np
import copy
import os 
import pathlib
import re
import time
import sys

repo_path=pathlib.Path(os.path.abspath(os.path.dirname(__file__)).__str__()).parents[0].absolute().__str__()
sys.path.insert(1,pathlib.Path(repo_path).absolute().__str__()) # repository path


def get_strtimestamp_dict(tz,timestamp_utc=None,timestamp_local=None):
    """
    Converts a given timestamp to a specific timezone and returns a dictionary with the UTC and local timestamps.

    Parameters:
    tz (str): The timezone to which the timestamp should be converted.
    timestamp_utc (str, optional): The timestamp in UTC. If this is not provided, timestamp_local must be provided.
    timestamp_local (str, optional): The timestamp in the local timezone. If this is not provided, timestamp_utc must be provided.

    Returns:
    dict: A dictionary with two keys: 'timestamp_utc' and 'timestamp_local'. The values are the UTC and local timestamps respectively, formatted as strings in the format 'YYYY-MM-DDTHH:MM:SSZ'.
    """

    if timestamp_utc is None:
        timestamp=pd.Timestamp(timestamp_local)
        localize_tz=tz
    else:    
        timestamp=pd.Timestamp(timestamp_utc)
        localize_tz="UTC"
     
    if timestamp.tz is None:
        timestamp_local=timestamp.tz_localize(localize_tz).tz_convert(tz).strftime("%Y-%m-%dT%H:%M:%S%z")
        timestamp_utc=timestamp.tz_localize(localize_tz).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S%z")
    else:
        timestamp_local=timestamp.tz_convert(tz).strftime("%Y-%m-%dT%H:%M:%S%z")
        timestamp_utc=timestamp.tz_convert(tz).strftime("%Y-%m-%dT%H:%M:%S%z")

    return {"timestamp_utc":timestamp_utc,"timestamp_local":timestamp_local}

def get_time_grid(start_time_utc,end_time_utc,tz,Ts):
    """
    Generates a grid of timestamps between a start and end time, with a specified frequency.

    Parameters:
    start_time_utc (str): The start time in UTC, formatted as a string.
    end_time_utc (str): The end time in UTC, formatted as a string.
    tz (str): The timezone to which the timestamps should be converted.
    Ts (int): The frequency of the timestamps in the grid, in seconds.

    Returns:
    pandas.DatetimeIndex: A DatetimeIndex object with timestamps between the start and end time, with frequency Ts, converted to the specified timezone.
    """
    start_time_dict=get_strtimestamp_dict(tz=tz,timestamp_utc=start_time_utc,timestamp_local=None)
    end_time_dict=get_strtimestamp_dict(tz=tz,timestamp_utc=end_time_utc,timestamp_local=None)
    start_time_utc=start_time_dict['timestamp_utc']
    end_time_utc=end_time_dict['timestamp_utc']

    n_data=int((pd.Timestamp(end_time_utc)-pd.Timestamp(start_time_utc))/pd.Timedelta(f"{int(Ts)} sec"))
    time_grid=pd.date_range(pd.Timestamp(start_time_utc), periods=n_data, freq=pd.Timedelta(f"{int(Ts)} sec"))
    time_grid=time_grid.tz_convert(tz)

    return time_grid

def add_time_grid(df,start_time,end_time,tz,Ts,timestamp_col='timestamp_local'):
    '''
    df dataframe
    start_time: str
    end_time: str
    tz: str
    Ts_raw: int
    '''

    if not(isinstance(df[timestamp_col].iloc[0], pd.Timestamp)):
        df[timestamp_col]=pd.to_datetime(df[timestamp_col])

    df=df.drop_duplicates(subset=[timestamp_col])
    tz_check=pd.Timestamp(df[timestamp_col].iloc[0])

    if tz_check.tz is None:
        # no timezone info in original data column. it will be local time
        tz_raw=tz
        df[timestamp_col]=df[timestamp_col].dt.tz_localize(tz_raw)
    else:
        # timezone info in original data column. all follows the tz info
        tz_raw=tz_check.tz

    start_time_dict=get_strtimestamp_dict(tz=tz_raw,timestamp_utc=None,timestamp_local=start_time)
    end_time_dict=get_strtimestamp_dict(tz=tz_raw,timestamp_utc=None,timestamp_local=end_time)
    start_time_utc=start_time_dict['timestamp_utc']
    end_time_utc=end_time_dict['timestamp_utc']

        
    
    # converted utc grid to tz grid
    time_grid=get_time_grid(start_time_utc=start_time_utc,end_time_utc=end_time_utc,tz=tz,Ts=Ts)


    df_base=pd.DataFrame(data={timestamp_col:time_grid})
    df=pd.merge(df_base,df,how='left',on=[timestamp_col])

    return df

def convert_to_utc(timestamp, tz):
    if timestamp is None:
        raise ValueError("timestamp should not be None.")
    
    if isinstance(timestamp, str):
        try:
            timestamp = pd.Timestamp(timestamp)
        except Exception as e:
            raise ValueError(f"timestamp_local can't be convertied by pd.Timestamp. {timestamp}")
        
        if timestamp.tzinfo is None:
            
            timestamp = pd.Timestamp(timestamp, tz=tz)
            print(f"timestamp_local is not timezone aware. It is assumed to be {tz}.")

    elif isinstance(timestamp, pd.Timestamp):
        timestamp = timestamp
    else:
        raise TypeError("timestamp_local should be a string or a pd.Timestamp instance.")
    
    timestamp_utc = timestamp.tz_convert('UTC')
    return timestamp_utc.strftime("%Y-%m-%d %H:%M:%S%z")


if __name__ == "__main__":
    
    tz="Asia/Seoul"
    Ts=60
    start_time_utc='2023-10-01 00:00:00+0000'
    end_time_utc='2023-10-01 01:00:00+0000'

    time_grid_test=get_time_grid(start_time_utc=start_time_utc,
                    end_time_utc=end_time_utc,
                    tz=tz,Ts=Ts)
    
    print(time_grid_test)

    print(pd.Timestamp(time_grid_test[0]))

    df_test=pd.DataFrame(data={'timestamp':time_grid_test})
    df_test['value']=1.0

    df_out=add_time_grid(df=df_test,start_time='2023-09-30 00:00:00+0000',
                    end_time='2023-10-02 01:00:00+0000',
                    tz='Asia/Seoul',Ts=60,timestamp_col='timestamp')
    

    print(df_out)

    df_test['timestamp']=df_test['timestamp'].dt.strftime("%Y-%m-%dT%H:%M:%S%z")

    df_out=add_time_grid(df=df_test,start_time='2023-09-30 00:00:00+0000',
                    end_time='2023-10-02 01:00:00+0000',
                    tz='Asia/Seoul',Ts=60,timestamp_col='timestamp')
    

    print(df_out)
    pass
    


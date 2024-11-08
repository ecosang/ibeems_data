

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
import gc
from multiprocessing.dummy import Pool

repo_path=pathlib.Path(os.path.abspath(os.path.dirname(__file__)).__str__()).parents[0].absolute().__str__()
sys.path.insert(1,pathlib.Path(repo_path).absolute().__str__()) # repository path

from database.data_formatter import df_history_parser
from database.query_raw_data import raw_data_query


query_split_days=2



def get_df_history(start_time_utc,end_time_utc,init_config,variable_map,credential):

    Ts_raw=init_config['Ts_raw']
    tz=init_config['tz']
    #zone_id_list=init_config['zone_id_list']

    variable_map=variable_map[variable_map['required']].reset_index(drop=True)
    if "n_thread" in init_config.keys():
        n_thread=init_config['n_thread']
    else:
        n_thread=1
    
    zone_id_list=variable_map['zone_id'].unique()

    # weather query
    data_df_dict={}

    for zid in zone_id_list:
        variable_map_zone=variable_map[variable_map['zone_id']==zid].reset_index(drop=True)
        zone_path_list=variable_map_zone['path'].to_list()
        input_dict_list=[]
        for path in zone_path_list:
            
            input_dict_={"path":path,
                         "start_time_utc":start_time_utc,
                        "end_time_utc":end_time_utc,
                        "credential":credential,
                        "init_config":init_config,
                        "query_split_days":query_split_days,
                        "variable_map":variable_map,
                        "zone_id":zid}
            input_dict_list.append(input_dict_)
        if n_thread==1:
            rr=[single_query_point(idl) for idl in input_dict_list]
        else:
            try:
                with Pool(n_thread) as pool:
                    rr=pool.map_async(single_query_point, input_dict_list).get()
                pool.terminate()
                gc.collect()
                del pool
            except Exception as e:
                rr=[single_query_point(idl) for idl in input_dict_list]

        
        rr=[r for r in rr if r['raw_df'] is not None]
        if len(rr)==0:
            pass
        else:
            for i,sr in enumerate(rr):
                if i==0:
                    df_temp=copy.deepcopy(sr['raw_df'])
                else:

                    df_temp=pd.merge(df_temp,sr['raw_df'],how='left',on="timestamp_local")
        
            data_df=df_history_parser(data_df=df_temp,
                        end_time_utc=end_time_utc,
                        start_time_utc=start_time_utc,
                        init_config=init_config,
                        variable_map=variable_map_zone,
                        zone_id=zid)
            
            data_df_dict[zid]=data_df.drop(columns=['timestamp_utc'])
    
    df_all=pd.DataFrame(data={'timestamp_local':pd.date_range(start=start_time_utc,end=end_time_utc,freq=f'{Ts_raw}s').tz_convert(tz)})

    for i,k in enumerate(data_df_dict.keys()):
        df_all=pd.merge(df_all,copy.deepcopy(data_df_dict[k]),how='left',on="timestamp_local")
    


    return df_all


    

    
    
    

def single_query_point(input_dict):

    path=input_dict['path']
    start_time_utc=input_dict['start_time_utc']
    end_time_utc=input_dict['end_time_utc']
    credential=input_dict['credential']
    init_config=input_dict['init_config']
    query_split_days=input_dict['query_split_days']
    



    raw_df=raw_data_query(path=path,
                start_time_utc=start_time_utc,
                end_time_utc=end_time_utc,
                credential=credential,
                init_config=init_config,
                query_split_days=query_split_days)

    # drop duplicates within df_history

    
    
    return {"raw_df":raw_df}
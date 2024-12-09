# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:23:58 2023

@author: DESKTOP-320
"""

#%% librart import
import os
import pathlib
import pandas as pd
import numpy as np
import sys

repo_path=pathlib.Path(os.path.abspath(os.path.dirname(__file__)).__str__()).parents[0].absolute().__str__()
sys.path.insert(1,pathlib.Path(repo_path).absolute().__str__()) # repository path
os.chdir(repo_path)

from database.function_io import get_config,get_point_list
from database.query_raw_data import raw_data_query
from database.data_formatter import df_history_parser
from database.database_util import get_df_history
from database.function_ts import convert_to_utc

def raw_data_query(project_id, start_time, end_time, save_dir=None,save_data=False):
    
    """
    Query raw data for a given project ID, start time, and end time.
    
    Description:
        - This function queries raw data for a given project ID, start time, and end time.
        - The raw data is saved as a CSV file in the specified directory.

    Args:
        project_id (str): The ID of the project.
        start_time (str): The start time in local or UTC timezone, formatted as %Y-%m-%d %H:%M:%S%z (e.g., '2024-03-07 00:00:00+0900' or '2024-03-07 00:00:00+0000')
        end_time (str): The end time in local or UTC timezone, formatted as %Y-%m-%d %H:%M:%S%z (e.g., '2024-03-07 00:00:00+0900' or '2024-03-07 00:00:00+0000')
        save_dir (str, optional): The directory to save the raw data. If not provided, the default save directory(download/{project_id}/raw_data_date_info.csv) will be used.
        save_data (bool): wheter to save the data or not
    Returns:
        dataframe, also data is stored in save_dir
    """
    
    init_path = f'data/{project_id}/init.yaml'

    init_config = get_config(config_path=init_path, config_format="yaml")
    tz = init_config['tz']
    start_time_utc = convert_to_utc(start_time, tz)
    end_time_utc = convert_to_utc(end_time, tz)

    start_time_local = pd.Timestamp(start_time_utc).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S%z")
    end_time_local = pd.Timestamp(end_time_utc).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S%z")

    credential = get_config(config_path=init_config['credential_path'], config_format="yaml")

    df_point_list = get_point_list(init_config=init_config, df_type="point_list")
    #df_point_list['available']=df_point_list['available'].astype('str')
    #df_point_list['required']=df_point_list['required'].astype('str')
    df_point_list['available']=df_point_list['available']=='True'
    df_point_list['required']=df_point_list['required']=='True'

    df_point_list = df_point_list[(df_point_list['required']) & (df_point_list['available'])].reset_index(drop=True)

    df_all = get_df_history(start_time_utc=start_time_utc,
                            end_time_utc=end_time_utc,
                            init_config=init_config,
                            variable_map=df_point_list,
                            credential=credential)
    if save_data:
        if save_dir is None:
            print(f"default save_dir is download/rawdata/{project_id}")
            save_path = pathlib.Path(f'download/rawdata/{project_id}/raw_data_{pd.Timestamp(start_time_local).strftime("%Y-%m-%d")}_{pd.Timestamp(end_time_local).strftime("%Y-%m-%d")}.csv')
        else:
            save_path = pathlib.Path(save_dir).joinpath(f'raw_data_{pd.Timestamp(start_time_local).strftime("%Y-%m-%d")}_{pd.Timestamp(end_time_local).strftime("%Y-%m-%d")}.csv')

        if pathlib.Path(save_path).parents[0].is_dir():
            pass
        else:
            pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            print(f'directory {str(pathlib.Path(save_path).parent)} is created')

        df_all.to_csv(save_path)
        print(f'data is stored in {save_path}')
    return df_all
    

if __name__=="__main__":
    import pathlib
    project_id='lotte_department'#'lotte_mart' or 'hdc' 'lotte_department'
    start_time_list=['2024-12-01 00:00:00+0900',]
                     #'2024-11-01 00:00:00+0900',
                     #'2024-02-01 00:00:00+0900',
                    #'2024-03-01 00:00:00+0900',
                    #  '2024-09-26 00:00:00+0900',
                    #  '2024-09-27 00:00:00+0900',
                     #]
    end_time_list=['2024-12-02 00:00:00+0900',]
                   #'2024-11-22 00:00:00+0900',
                    #'2024-03-01 00:00:00+0900',
                    #'2024-04-01 00:00:00+0900',
                    #  '2024-09-27 00:00:00+0900',
                    #  '2024-09-28 00:00:00+0900',
                     #]
    #start_time='2024-09-03 00:00:00+0900'
    #end_time='2024-09-04 00:00:00+0900'
    
    for start_time,end_time in zip(start_time_list,end_time_list):
        save_dir=None

        df_all=raw_data_query(project_id,start_time,end_time,save_dir=None)
        print(df_all)
        download_folder=f"download/{project_id}"
        if pathlib.Path(download_folder).is_dir():
            pass
        else:
            pathlib.Path(download_folder).mkdir(parents=True, exist_ok=True)
            print(f'directory {download_folder} is created')
                  
        df_all.to_csv(pathlib.Path(download_folder).joinpath(f"df_{project_id}_{pd.Timestamp(start_time).strftime('%Y-%m-%d')}_{pd.Timestamp(end_time).strftime('%Y-%m-%d')}.csv"))

# %%

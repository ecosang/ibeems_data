# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:23:58 2023

@author: DESKTOP-320
"""

#%% librart import
import os
import json
import requests
#import pymysql
import pathlib
import pandas as pd
import numpy as np
import copy
import sys
import yaml

repo_path=pathlib.Path(os.path.abspath(os.path.dirname(__file__)).__str__()).parents[1].absolute().__str__()
sys.path.insert(1,pathlib.Path(repo_path).absolute().__str__()) # repository path
os.chdir(repo_path)


from database.function_io import get_config,get_point_list
from database.function_ts import convert_to_utc
from database.query_raw_data import raw_data_query

def check_data_exist(project_id,start_time,end_time,update_metadata=False):
    """
    Check the data existance for a given project.
    Description:
        - This function checks the data existance for the given project and time period.
        - The data existance is checked based on the point list and the raw data.
        - The result is saved as a csv file in the data/project_id folder.
        

    Args:
        project_id (str): The ID of the project. #'lotte_mart' or 'hdc'
        start_time (str): The start time in local or UTC timezone, formatted as %Y-%m-%d %H:%M:%S%z (e.g., '2024-03-07 00:00:00+0900' or '2024-03-07 00:00:00+0000')
        end_time (str): The end time in local or UTC timezone, formatted as %Y-%m-%d %H:%M:%S%z (e.g., '2024-03-07 00:00:00+0900' or '2024-03-07 00:00:00+0000')
        update_metadata (bool, optional): Whether to update the metadata. Defaults to False.

    
    Returns:
        None
    """
    init_path=f'data/{project_id}/init.yaml'

    init_config=get_config(config_path=init_path,config_format="yaml")

    tz=init_config['tz']
    start_time_utc=convert_to_utc(start_time,tz)
    end_time_utc=convert_to_utc(end_time,tz)
    
    start_time_local=pd.Timestamp(start_time_utc).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S%z")
    end_time_local=pd.Timestamp(end_time_utc).tz_convert(tz).strftime("%Y-%m-%d %H:%M:%S%z")
    
    credential=get_config(config_path=init_config['credential_path'],config_format="yaml")

    sheet_name_list=init_config['point_list_path']
    
    for sn in sheet_name_list:
        output_path=pathlib.Path(f"data/{project_id}").joinpath(sn['file_name']+f".{sn['format']}")
        
        df_point_list=pd.read_csv(output_path)
        
        df_point_list['data_exist']=False

        point_dict_list=[row.to_dict() for ix,row in df_point_list.iterrows()]


        pdl=copy.deepcopy(point_dict_list)

        for i,pl in enumerate(point_dict_list):
            if pl['required']:
                raw_df=raw_data_query(path=pl['path'],
                            start_time_utc=start_time_utc,
                            end_time_utc=end_time_utc,
                            credential=credential,
                            init_config=init_config,
                            query_split_days=2)
                
                if raw_df is not None:
                    print(f'!!!!!!!!!!!!!!!!!!{str(pl["object_name"])+str(pl["object_id"])} exist!!!!!!!!!!!!!!!!!!!!')
                    pdl[i]['data_exist']=True
                else:
                    print(f'{str(pl["object_name"])+str(pl["object_id"])} does not exist')
                    pdl[i]['data_exist']=False
            else:
                print(f'skip {str(pl["object_name"])+str(pl["object_id"])}')
                pdl[i]['data_exist']=False

        df=pd.DataFrame(pdl)
        
        if update_metadata:
            df.to_csv(output_path,encoding='utf-8-sig')

        else:
            
            save_path=pathlib.Path(f"data/{project_id}/check_data").joinpath(sn['file_name']+f"_{pd.Timestamp(start_time_local).strftime('%Y%m%d')}_{pd.Timestamp(end_time_local).strftime('%Y%m%d')}.csv")
            
            if pathlib.Path(save_path).parents[0].is_dir():
                pass
            else:
                pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(save_path,encoding='utf-8-sig')


if __name__=="__main__":

    '''
    This file check and update the currently available data. 

    '''
    project_id='lotte_department' # lotte_mart, hdc
    start_time="2024-12-01 00:00:00+0900"
    end_time="2024-12-01 23:59:00+0900"
    check_data_exist(project_id,start_time,end_time,update_metadata=False)

    # project_id='hdc' # lotte_mart, hdc
    # init_path=f'data/{project_id}/init.yaml'
    # update_metadata=False

    # start_time_local="2024-03-18 00:00:00+0900"
    # end_time_local="2024-03-18 23:59:00+0900"

    # start_time_utc=pd.Timestamp(start_time_local).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S%z")
    # end_time_utc=pd.Timestamp(end_time_local).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S%z")



    # init_config=get_config(config_path=init_path,config_format="yaml")

    # credential=get_config(config_path=init_config['credential_path'],config_format="yaml")

    # sheet_name_list=init_config['point_list_path']
    # metadata_rawdata_path=credential['metadata_rawdata_path']
    # # sn['file_name']+f".{sn['format']}

    # for sn in sheet_name_list:
    #     output_path=pathlib.Path(f"data/{project_id}").joinpath(sn['file_name']+f".{sn['format']}")
        
    #     df_point_list=pd.read_csv(output_path)
        
    #     df_point_list['data_exist']=False

    #     point_dict_list=[row.to_dict() for ix,row in df_point_list.iterrows()]


    #     pdl=copy.deepcopy(point_dict_list)

    #     for i,pl in enumerate(point_dict_list):
    #         if pl['required']:
    #             raw_df=raw_data_query(path=pl['path'],
    #                         start_time_utc=start_time_utc,
    #                         end_time_utc=end_time_utc,
    #                         credential=credential,
    #                         init_config=init_config,
    #                         query_split_days=2)
                
    #             if raw_df is not None:
    #                 print(f'!!!!!!!!!!!!!!!!!!{str(pl["object_name"])+str(pl["object_id"])} exist!!!!!!!!!!!!!!!!!!!!')
    #                 pdl[i]['data_exist']=True
    #             else:
    #                 print(f'{str(pl["object_name"])+str(pl["object_id"])} does not exist')
    #                 pdl[i]['data_exist']=False
    #         else:
    #             print(f'skip {str(pl["object_name"])+str(pl["object_id"])}')
    #             pdl[i]['data_exist']=False

    #     df=pd.DataFrame(pdl)
        
    #     if update_metadata:
    #         df.to_csv(output_path,encoding='utf-8-sig')

    #     else:
    #         save_path=pathlib.Path(f"data/{project_id}").joinpath(sn+f"_{pd.Timestamp(start_time_local).strftime('%Y%m%d')}_{pd.Timestamp(end_time_local).strftime('%Y%m%d')}.{init_config['point_list_path']['format']}")

    #         df.to_csv(save_path,encoding='utf-8-sig')



    # #df_point_list = get_point_list(init_config=init_config,df_type="point_list")
    
    # # zone_id	variable_name	raw_unit	type	required	available
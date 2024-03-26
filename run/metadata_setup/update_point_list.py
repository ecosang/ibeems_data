
import os
import pathlib
import pandas as pd
import numpy as np
import sys

repo_path=pathlib.Path(os.path.abspath(os.path.dirname(__file__)).__str__()).parents[1].absolute().__str__()
sys.path.insert(1,pathlib.Path(repo_path).absolute().__str__()) # repository path
os.chdir(repo_path)
print(repo_path)

from database.function_io import get_config,get_point_list
from database.query_raw_data import raw_data_query
from database.data_formatter import df_history_parser
from database.database_util import get_df_history

def update_point_list(project_id, update_path=False):
    """
    Update the point list for a given project.

    Args:
        project_id (str): The ID of the project. #'lotte_mart' or 'hdc'
        update_path (bool, optional): Whether to update the path (오브젝트경로) when there is changes in Google spreadsheet(오브젝트경로). Defaults to False.

    Description:
        - This function reads the raw metadata from the Google spreadsheet (credential['metadata_rawdata_path']) and creates a metadata file for each sheet name in the init_config file.
        - The metadata file is saved as {sheet_name}.csv in the data/project_id folder.
        - If update_path is True, the function will update the file by running the code again after updating the Google spreadsheet.
        
    Returns:
        None
    """
    
    init_path=f'data/{project_id}/init.yaml'

    init_config=get_config(config_path=init_path,config_format="yaml")
    credential=get_config(config_path=init_config['credential_path'],config_format="yaml")

    sheet_name_list=init_config['point_list_path']
    metadata_rawdata_path=credential['metadata_rawdata_path']

    for sn in sheet_name_list:
        df=pd.read_excel(metadata_rawdata_path,sheet_name=sn['file_name'],header=None)
        df=df.dropna(how='all')
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        df = df.reset_index(drop=True)
        df=df.rename(columns=sn['raw_column_map'])
        if "comment" not in df.columns:
            df['comment']=np.nan
        df=df.loc[:,['path','object_id','object_name','comment']].reset_index(drop=True)

        output_path=pathlib.Path(f"data/{project_id}").joinpath(sn['file_name']+f".{sn['format']}")
        output_backup_path=pathlib.Path(f"data/{project_id}").joinpath(sn['file_name']+f"_backup.{sn['format']}")
        
        if pathlib.Path(output_path).is_file():
            df_=pd.read_csv(output_path)
            # make columns (dummy columns for consistency.)
            for cn in sn['columns']:
                if cn not in df_.columns:
                    df_[cn]=np.nan
            df_.to_csv(output_backup_path, encoding="utf-8-sig",index=False)
        else:
            df_=None

        # format df_ and join to the object_id,object_name
        if df_ is None:
            for cn in sn['columns']:
                if cn not in df.columns:
                    df[cn]=np.nan
        else:
            # update df
            if update_path:
                df=pd.merge(df,df_.drop(columns=['path','comment']),how='left',on=['object_id','object_name'])
            else:
                df=pd.merge(df,df_.drop(columns=['comment']),how='left',on=['object_id','object_name','path'])
        # save xxx.csv for metadata.
        if pathlib.Path(output_path).parents[0].is_dir():
            pass
        else:
            pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"{project_id}'s {sn['file_name']} sheet is stored in {output_path}.")
        print(f"original columns are {sn['raw_column_map']}")
        df.to_csv(output_path, encoding="utf-8-sig",index=False)


if __name__=="__main__":
    
    
    project_id='lotte_mart'
    update_path=False
    update_point_list(project_id, update_path=update_path)
    

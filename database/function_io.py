import yaml
import re
import json
import pathlib
import pandas as pd
import numpy as np

def get_point_list(init_config,df_type="point_list"):

    project_id=init_config['project_id']
    

    if (f'{df_type}_path' in init_config.keys()) or (f'{df_type}' in init_config.keys()):
        if f'{df_type}_path' in init_config.keys():
            file_info_list=init_config[f'{df_type}_path']
        else:
            file_info_list=init_config[f'{df_type}']
    else:
        raise ValueError(f'{df_type}_path or {df_type} not in init_config.keys() there are {init_config.keys()}')
    
    df_list=[]
    for file_info in file_info_list:

        if file_info['format']=="csv" or file_info['format']=="url_csv":
            output_path=pathlib.Path(f"data/{project_id}").joinpath(file_info['file_name']+f".csv")

            try:
                df=pd.read_csv(output_path,index_col=False)
                df=df[file_info['columns']]
                df_list.append(df)
            except Exception as e:
                print(f'file not found '+str(output_path))
                print(f"error: {e}")
        else:
            raise ValueError(f'csv or csv_url only. {file_info}')
    if len(df_list)==0:
        df=None
    else:
        df=pd.concat(df_list).reset_index(drop=True)
    
    return df


def get_config(config_path,config_format="yaml"):
    if config_format=="yaml":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_format=="json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError("json, yaml only.")
        logger.error(f'json, yaml only. get_config function check {config_path} and {config_format}')
    return config

def get_config_type(input_path):
    
    if input_path == "None" or input_path is None:
        config_type=None
    elif re.search("yaml",input_path):
        config_type="yaml"
    elif re.search("json",input_path):
        config_type='json'
    else:
        raise ValueError("only yaml and json are available.")
        logger.error(f'only yaml and json are available. get_config_type function {input_path}')
    return config_type

# def get_credential(credential_path):
#     config_type=get_config_type(credential_path)
    
#     with open(credential_path,"r") as f:
#         if config_type=="json":
#             credential=json.load(f)
#         elif config_type=="yaml":
#             credential=yaml.safe_load(f)
#     return credential



# https://stackoverflow.com/questions/34667108/ignore-dates-and-times-while-parsing-yaml
yaml.SafeLoader.yaml_implicit_resolvers = {
    k: [r for r in v if r[0] != 'tag:yaml.org,2002:timestamp'] for
    k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}



class NpEncoder(json.JSONEncoder):
    # https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
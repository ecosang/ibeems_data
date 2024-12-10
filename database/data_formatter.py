import pandas as pd
import ast
import numpy as np
import re


def make_monotonic(vec):
    
    vec_diff=np.diff(vec)
    if np.any(vec_diff<-1e7):
        change_point_idx=np.where(vec_diff<-1e7)[0][0]
        #print(change_point_idx)
        vec[(change_point_idx+1):]=vec[(change_point_idx+1):]+vec[change_point_idx]
        
    for i in range(1, len(vec)):
        if vec[i] < vec[i-1]:
            vec[i] = vec[i-1]
    return vec




def df_history_parser(data_df,end_time_utc,start_time_utc,init_config,variable_map,zone_id=None):
    
    data_df=data_df.copy()
    dict_columns=get_dict_columns(variable_map[(variable_map['zone_id']==zone_id) &(variable_map['required'])].reset_index(drop=True))

    Ts_raw=init_config['Ts_raw']
    tz=init_config['tz']
    
    #interpolation_limit=int(3600/Ts_raw*history_interpolation_hours) #4 hours  
    
    data_df['timestamp_utc']=data_df['timestamp_local'].dt.tz_convert("UTC")
    #data_df=data_df.sort_values('timestamp').reset_index(drop=True)
    #data_df=data_df.rename(columns={"timestamp":"timestamp_utc"})
    # this can accept both strign and timestamp.
    #data_df['timestamp_utc']=pd.to_datetime(data_df['timestamp_utc'],utc=True).round(pd.Timedelta(f'{int(Ts_raw)}sec'))
    
    if zone_id=="weather":
        check_variable="T_oa"
    else:
        var_list=variable_map['variable_name'].unique()
        if "T_ahu_ra" in var_list:
            check_variable="T_ahu_ra"#"T_csp"#"T_za"
        else:
            check_variable="T_ahu_csp"
    #ariable_map[variable_map['variable_name'].str.contains("T_oa")]
    if zone_id=="common" or zone_id=="weather":
        pass
    else:
        check_map=variable_map[((variable_map['variable_name'].str.contains(check_variable))&(variable_map['zone_id']==str(zone_id)))].reset_index(drop=True)
        if check_map.empty or (check_map['path'][0] not in data_df.columns):
            pass
        else:
            check_df=data_df[['timestamp_utc',check_map['path'][0]]].dropna().reset_index(drop=True)
            #.merge(check_map[['topic_name']],how='inner',on=['topic_name'])
            last_time=check_df['timestamp_utc'].iloc[-1]
            first_time=check_df['timestamp_utc'].iloc[0]
            #last_time=data_df[data_df['topic_name']==check_map['topic_name'][0]].sort_values('timestamp_utc').tail(1).reset_index(drop=True)['timestamp_utc'][0]
            #first_time=data_df[data_df['topic_name']==check_map['topic_name'][0]].sort_values('timestamp_utc').head(1).reset_index(drop=True)['timestamp_utc'][0]
            
            if pd.Timestamp(end_time_utc)<=last_time:
                last_time_delta=0
                last_time=pd.Timestamp(end_time_utc)
            else:
                last_time_delta=(pd.Timestamp(end_time_utc)-last_time).total_seconds()
            
            if pd.Timestamp(start_time_utc)>=first_time:
                start_time_utc_raw=start_time_utc
                start_time_utc=first_time.strftime("%Y-%m-%dT%H:%M:%S%z")
                history_start_time_utc_raw=pd.Timestamp(start_time_utc_raw).round(pd.Timedelta(f'{int(Ts_raw)}sec')).strftime("%Y-%m-%dT%H:%M:%S%z")
            else:
                history_start_time_utc_raw=pd.Timestamp(start_time_utc).round(pd.Timedelta(f'{int(Ts_raw)}sec')).strftime("%Y-%m-%dT%H:%M:%S%z")
        
            if last_time_delta>(Ts_raw*20):
                print(f'table_name is {zone_id}.')
                print(f"last time delta (query end time ({end_time_utc})-last data time({last_time})) is more than raw data sampling time ({Ts_raw}) *10. {last_time_delta}. Please double check ")
                print(data_df.sort_values('timestamp_local').tail())
            else:
                end_time_utc=last_time.strftime("%Y-%m-%dT%H:%M:%S%z")

    history_end_time_utc=pd.Timestamp(end_time_utc).strftime("%Y-%m-%dT%H:%M:%S%z")
    history_start_time_utc=pd.Timestamp(start_time_utc).round(pd.Timedelta(f'{int(Ts_raw)}sec')).strftime("%Y-%m-%dT%H:%M:%S%z")
    
    data_df=data_df[(data_df['timestamp_utc']>=pd.Timestamp(history_start_time_utc))&(data_df['timestamp_utc']<=pd.Timestamp(history_end_time_utc))]
    
    
    #df_base=get_time_grid(start_time,end_time,tz,Ts_raw,timestamp_col='timestamp_local')

    #n_data=int((pd.Timestamp(history_end_time_utc,tz="UTC")-pd.Timestamp(history_start_time_utc,tz="UTC"))/pd.Timedelta(f"{Ts_raw} sec"))
    #time_grid=pd.date_range(pd.Timestamp(history_start_time_utc,tz="UTC"), periods=n_data, freq=pd.Timedelta(f"{Ts_raw} sec"))
    #df_base=pd.DataFrame(data={"timestamp_utc":time_grid})
    
    rename_map=dict_columns['rename_map']
    unit_map=dict_columns['unit_map']
    type_map=dict_columns['type_map']
    
    #raw_cnames=data_df.columns
    #data_df=data_df[data_df['topic_name'].isin(dict_columns['topic_name'])].reset_index(drop=True)
    #data_df=data_df.pivot_table(index='timestamp_utc',columns='topic_name',values='value_string',aggfunc='first').reset_index()
    cnames=data_df.columns

    minute=data_df['timestamp_utc'].dt.minute.to_numpy()
    
    for cn in dict_columns['path']:
        if cn=="timestamp_utc" or cn=="timestamp" or cn=="time" or cn=="timestamp_local":
            pass
        elif cn in rename_map.keys():
            if cn in cnames:
                #print(cn)
                rn=rename_map[cn]
                un=unit_map[cn]
                tn=type_map[cn]

                data_df[cn]=get_single_value(data_df[cn].to_numpy(),unit_name=un,type_name=tn,minute=minute,var_name=cn)
                data_df=data_df.copy()
                if zone_id is None or zone_id=="weather" or zone_id=="common":
                    data_df=data_df.rename(columns={cn:f'{rn}'})
                else:    
                    data_df=data_df.rename(columns={cn:f'{rn}_{zone_id}'})
            else:
                # cn is not in cnames
                rn=rename_map[cn]
                if zone_id is None or zone_id=="weather" or zone_id=="common":
                    data_df[f'{rn}']=np.nan
                else:
                    data_df[f'{rn}_{zone_id}']=np.nan
        else:
            data_df=data_df.drop(columns=[cn])

    #df_base=pd.merge(df_base, data_df, how='left', on=['timestamp_utc'])
    
    
        
    
    if zone_id is None or zone_id=="weather" or zone_id=="common":
        pass
    else:
        pass
        # zone level formatting
        # # sp conversion
        # col_temp=[x for x in list(df_base.columns) if re.findall(f"i_hc_\d+_{zone_id}",x)]
        # if len(col_temp)>0:
        #     col_temp.sort()
        #     for j,cname in enumerate(col_temp):
        #         i_hc_temp=df_base[cname].to_numpy()
        #         sysmode_temp=df_base[f'sysmode_{zone_id}'].to_numpy()
        #         i_cool_temp=i_hc_temp*1*(sysmode_temp==3)
        #         i_heat_temp=i_hc_temp*1*(sysmode_temp==4)
        #         df_base[f'i_heat_{zone_id}_{j}']=i_heat_temp
        #         df_base[f'i_cool_{zone_id}_{j}']=i_cool_temp
        
        # #heatpump power
        # col_temp_power=[x for x in list(df_base.columns) if re.findall(f"P_hp_\d+_{zone_id}",x)]
        # if len(col_temp_power)>0:
        #     df_base=merge_plug_load(raw_df=df_base,zone_id=zone_id,target="hp",header="P")
        
    
    #df_base['timestamp_utc']=pd.DatetimeIndex(df_base['timestamp_utc'],tz="UTC")
    
    return data_df




def convert_missing_to_nan(vec,type_):
    vec=vec.astype('object')
    vec[vec=='null']=np.nan
    vec[vec=='NaN']=np.nan
    vec[vec=='None']=np.nan
    vec[vec==None]=np.nan
    vec[pd.isna(vec)]=np.nan
    if type_=="list" or type_=="list_ecobee":
        vec=vec.astype('float')
    elif type_=="string" or type_=="str":
        vec=vec.astype('str')
    else:
        vec=vec.astype(type_)
    return (vec)


def get_single_value(value,unit_name,type_name,minute=None,var_name=None):
    
    if type_name=="list_ecobee":
        #
        if minute is None:
            raise ValueError(f"minute is not given for list_ecobee {value}, {unit_name}, {type_name}")
        
        minute_index=minute%15//5 # 0, 1, 2 index for list  [180,300,300]
        value=[str(va) for va in value]
        value=np.array([ast.literal_eval(value[i])[minute_index[i]] if (value[i][0]=='[') else np.nan for i in np.arange(minute.shape[0])])/300
    
    if type_name=="list":
        # first value of ecobee
        value=[str(va) for va in value]
        value=np.array([ast.literal_eval(va)[0] if (va[0]=='[') else np.nan for va in value])
        
    value=convert_missing_to_nan(value,type_name)    
    
    if unit_name=="F":
        value=(value-32)/1.8
    elif unit_name=="C":       
        pass
    elif unit_name=="K":       
        value=value-273.15
    elif unit_name=="W_m2" or unit_name=="W/m2" or unit_name=="W":
        value=(value)/1000 #W/m2 to kW/m2
    elif unit_name=="CMH" or unit_name=='cmh':
        value=(value)/3600 #CMH to m3/s
    elif unit_name=="CFM":
        value=(value)*0.00047194745 #CFM to m3/s
    elif unit_name=="lpm":
        value=(value)/60000
    elif unit_name=="100%" or unit_name=="%":
        value=(value)/100
    elif unit_name=="10F":
        value=((value)/10-32)/1.8
    elif unit_name=="cum_30" or unit_name=="cum_15" or unit_name=="cum_60":
        # cumulative value by 30seconds..
        if unit_name=="cum_30":
            rolling_average=30
        elif unit_name=="cum_15":
            rolling_average=15
        else:
            rolling_average=60
        value=make_monotonic(value)
    
        value=np.concatenate([np.array([0]),np.diff(value)])/1000/30
        
        df_value=pd.DataFrame(data={'value':value})
        value=df_value['value'].rolling(rolling_average,min_periods=1,center=True).mean().to_numpy()
    
    elif unit_name=="cum_mv30_60/1000":
        # cumulative value by 30seconds.. and divided by 1000 due to scale
        rolling_average=30
        Ts_=60
        value=make_monotonic(value)
        value=np.concatenate([np.array([0]),np.diff(value)])/1000/Ts_
        df_value=pd.DataFrame(data={'value':value})
        value=df_value['value'].rolling(rolling_average,min_periods=1,center=True).mean().to_numpy()
    elif unit_name=="cum_mv30_60/60":
        # cumulative value by 30seconds.. and divided by 60 due to scale
        rolling_average=30
        Ts_=60
        value=make_monotonic(value)
        value=np.concatenate([np.array([0]),np.diff(value)])/60/Ts_
        df_value=pd.DataFrame(data={'value':value})
        value=df_value['value'].rolling(rolling_average,min_periods=1,center=True).mean().to_numpy()
        
    elif unit_name=="cum_mv30_60":
        # cumulative value by 30seconds.. and divided by 1000 due to scale
        rolling_average=30
        Ts_=60
        value=make_monotonic(value)
        value=np.concatenate([np.array([0]),np.diff(value)])/Ts_
        df_value=pd.DataFrame(data={'value':value})
        value=df_value['value'].rolling(rolling_average,min_periods=1,center=True).mean().to_numpy()
    elif unit_name=="cum_mv30_3600/60":
        # cumulative value by 30seconds.. and divided by 1000 due to scale
        rolling_average=30
        Ts_=60
        value=make_monotonic(value)
        value=np.concatenate([np.array([0]),np.diff(value)])*(3600/Ts_)
        df_value=pd.DataFrame(data={'value':value})
        value=df_value['value'].rolling(rolling_average,min_periods=1,center=True).mean().to_numpy()

    return value


def get_dict_columns(variable_map): # ,table_name="cel_hvac"
    map_csv=variable_map
    path_name=map_csv['path'].to_list()
    variable_name=map_csv['variable_name'].to_list()
    unit_name=map_csv['raw_unit'].to_list()
    type_name=map_csv['type'].to_list()
    dict_columns={}
    rename_map={}
    unit_map={}
    type_map={}
    for tn,rn,un,tp in zip(path_name,variable_name,unit_name,type_name):
        rename_map[tn]=rn
        unit_map[tn]=un
        type_map[tn]=tp
    dict_columns['rename_map']=rename_map
    dict_columns['unit_map']=unit_map
    dict_columns['type_map']=type_map
    dict_columns['path']=path_name
    return dict_columns

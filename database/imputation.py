
import numpy as np
import pandas as pd
import copy
from collections import Counter


"""
Imputation module: This module contains the Imputation class, which is used to impute missing data in a time series.

Mode labeling module: This module contains the ModeLabeling class, which is used to label the modes of a time series.

TODO
add filter mode data. if it is not in specific mode categories, just remove it.

"""

class Imputation:
    def __init__(self,array,interpolation_method='ffill',interpolation_limit=3):
        self.array=array
        self.imputed=np.repeat(False,array.shape[0])
        self.interpolation_method=interpolation_method
        self.interpolation_limit=interpolation_limit
        pass
    def interpolate(self,array,interpolation_method,interpolation_limit):
        if interpolation_method =="nan":
            # doing nothing
            pass
        else:    
            # impute with method and limit
            array_original=copy.deepcopy(array)
            #array=array.astype("float")
            if interpolation_method =="linear":
                limit_direction="both"
            else:
                limit_direction=None
            if interpolation_method=="bfill":
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.bfill.html
                # TODO https://stackoverflow.com/questions/77900971/pandas-futurewarning-downcasting-object-dtype-arrays-on-fillna-ffill-bfill
                array=array.astype(object)
                unique_values = np.unique(array).tolist()
                if any(val in unique_values for val in ["False", "True"]):
                    array[array=="False"]=0.0
                    array[array=="True"]=1.0

                array=array.astype(float)

                array=(pd.Series(array).bfill(limit=interpolation_limit)).to_numpy()
            elif interpolation_method=="ffill":
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.ffill.html
                # TODO https://stackoverflow.com/questions/77900971/pandas-futurewarning-downcasting-object-dtype-arrays-on-fillna-ffill-bfill
                # fillna of object is not allowed?
                array=array.astype(object)
                unique_values = np.unique(array).tolist()
                if any(val in unique_values for val in ["False", "True"]):
                    array[array=="False"]=0.0
                    array[array=="True"]=1.0

                array=array.astype(float)

                array=(pd.Series(array).ffill(limit=interpolation_limit)).to_numpy()
            else:
                # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.interpolate.html
                
                array=pd.Series(array).interpolate(method=interpolation_method,limit=interpolation_limit,limit_direction=limit_direction).to_numpy()
            
            imputed=copy.deepcopy(self.imputed)
            imputed[array_original!=array]=True # so that previous already "Ture" stay there.
            self.imputed=imputed
        
        return array
    def replace_array(self,array):
        """
        Replace array in case to recycle the object for new data array (instead of createing new Imputation())
        """
        self.array=array
        self.imputed=np.repeat(False,array.shape[0])

    def interpolation_param(self,interpolation_method,interpolation_limit):
        
        if interpolation_method == "NaN" or interpolation_method == "nan":
            interpolation_method="nan"
        elif interpolation_method is None:
            interpolation_method=self.interpolation_method
        else:
            interpolation_method=interpolation_method

        if interpolation_limit is None or interpolation_limit=="None":
            
        
            interpolation_limit=None # unlimited
        elif interpolation_limit=="inf" or interpolation_limit=="Inf" or interpolation_limit == np.inf:
            interpolation_limit=None #unlimited
        
        return interpolation_method,interpolation_limit
    
    def impute_change(self, th=3,th_len=10,interpolation_method=None,interpolation_limit=None):
        """
        This function imputes changes in the data that exceed a certain threshold.

        Parameters:
        th (int): The threshold for detecting significant changes. Default is 3.
        method (str): The method to use for imputation. If "NaN" or "nan", it will replace with NaN. 
                    If None, it will use the method stored in self.method. Default is None.
        limit (int): The maximum number of consecutive NaN values to fill. If None, it will use the limit stored in self.limit. Default is None.
        len (int): detect errors by len numbers
        Returns:
        None. It modifies the self.array in-place.
        """

        interpolation_method,interpolation_limit=self.interpolation_param(interpolation_method,interpolation_limit)
        

        array=copy.deepcopy(self.array)
        prev_val=np.nan 

        # 1. check value change more than th
        # 2. if change detected put np.nan for the value, check the next value
        # 3. if the next value is also change more than th, keep put np.nan for the next value unless it is a new sequence (by checking the next next value)
        # 4. if the next value is not change more than th, stop detection.
        len_count=0
        for i in range(len(array) - 1):
            
            if np.isnan(prev_val):
                if np.isnan(array[i]):
                    continue
                elif np.abs(array[i+1] - array[i]) > th:
                    prev_val = array[i]
                    array[i+1] = np.nan
                    len_count=len_count+1
            elif not np.isnan(array[i+1]) and np.abs(array[i+1] - prev_val) > th:
                len_count=len_count+1
                # detection continues
                if len_count>th_len:#and np.abs(prev_val - array[i+2]) <= th:
                    # stop detection
                    prev_val = np.nan
                    len_count=0
                else:
                    prev_val = prev_val# if (i+2) < len(array) and np.abs(prev_val - array[i+2]) > th else np.nan
                    array[i+1] = np.nan
                    len_count=len_count+1
                    
            else:
                prev_val = np.nan


        self.array=self.interpolate(array=array,interpolation_method=interpolation_method,interpolation_limit=interpolation_limit)
        
    def impute(self,interpolation_method=None,interpolation_limit=None):
        interpolation_method,interpolation_limit=self.interpolation_param(interpolation_method,interpolation_limit)
        array=copy.deepcopy(self.array)
        self.array=self.interpolate(array=array,interpolation_method=interpolation_method,interpolation_limit=interpolation_limit)
        
    def impute_minmax(self,min_val,max_val,interpolation_method=None,interpolation_limit=None):
        
        interpolation_method,interpolation_limit=self.interpolation_param(interpolation_method,interpolation_limit)
        

        array=copy.deepcopy(self.array)
        array=array.astype("float")
        #array=self.interpolate(array=array,method=method,limit=limit)

        erroneous_indices = np.where((array < min_val) | (array > max_val))[0]
        array[erroneous_indices] = np.nan
        self.array=self.interpolate(array=array,interpolation_method=interpolation_method,interpolation_limit=interpolation_limit)

    def impute_zscore(self, zscore=3,interpolation_method=None,interpolation_limit=None):
        '''
        zscore: within the data, data outside of mean(+-)zscore*std will be removed.
        '''
        interpolation_method,interpolation_limit=self.interpolation_param(interpolation_method,interpolation_limit)
        
        array=copy.deepcopy(self.array)
        array=array.astype("float")
        # Calculate the mean and standard deviation of the data
        mean = np.nanmean(array)
        std = np.nanstd(array)

        # Define the threshold for detecting outliers
        lower_threshold = mean - zscore * std
        upper_threshold = mean + zscore * std

        # Identify the indices of the outliers
        outliers = np.where((array < lower_threshold) | (array > upper_threshold))[0]

        # Replace the outliers with numpy nan
        array[outliers] = np.nan

        self.array=self.interpolate(array=array,interpolation_method=interpolation_method,interpolation_limit=interpolation_limit)
    
    def impute_medianfilter(self,window=3,std_level=1,interpolation_method=None,interpolation_limit=None):
        
        # https://stackoverflow.com/questions/62473007/modify-outliers-caused-by-sensor-failures-in-timeseries-data
        
        interpolation_method,interpolation_limit=self.interpolation_param(interpolation_method,interpolation_limit)
        
        array=copy.deepcopy(self.array)
        array=array.astype("float")
        array_series=pd.Series(array)
        array_med= array_series.rolling(window,center=True,min_periods=1).median().to_numpy()
        def std_percentile(x):
            x_th=np.nanpercentile(x,[0.025,99.75])
            x=x[(x>=x_th[0])&(x<=x_th[1])]
            return np.std(x)

        array_std = array_series.rolling(window,center=True,min_periods=1).apply(std_percentile,raw=True).to_numpy()
        outliers=(array>(array_med+array_std*std_level))|(array<(array_med-array_std*std_level))
        
        array[outliers] = np.nan
        self.array=self.interpolate(array=array,interpolation_method=interpolation_method,interpolation_limit=interpolation_limit)
    
    

    def impute_state(self, n_state=2,state_th=0,interpolation_method=None,interpolation_limit=None):
        '''
        n_state: minimum number of state that is valid, i.e., state <n_state is replaced by nan
        '''
        n_state=max(2,n_state)
        interpolation_method,interpolation_limit=self.interpolation_param(interpolation_method,interpolation_limit)

        raw_array=copy.deepcopy(self.array)
        raw_array=raw_array.astype("float")

        if (raw_array.dtype.type is np.str_) or (raw_array.dtype.type is np.object_):
            # need to string to numeric.
            string_array=True
            unique_string=np.unique(raw_array)
            mapping_dict={}
            unmapping_dict={}
            for i,us in enumerate(unique_string):
                if us=="nan":
                    pass
                else:
                    mapping_dict[us]=i
                    unmapping_dict[i]=us

            mapping_dict["nan"]=np.nan
            unmapping_dict[np.nan]="nan"

            array = np.array([mapping_dict[vs] for vs in raw_array])
        else:
            array=raw_array
            string_array=False
                
        array = array.astype("float")

        #rleid = np.cumsum(np.concatenate(([0], np.diff(array) != 0)))
        rleid = np.cumsum(np.concatenate(([0], np.abs(np.diff(array)) > state_th)))
        #print(rleid.astype('float'))
        unique_rleid = np.unique(rleid)
        
        for rleid_val in unique_rleid:
            indices = np.where(rleid == rleid_val)[0]
            if len(indices) <= n_state:
                array[indices] = np.nan
        array=self.interpolate(array=array,interpolation_method="ffill",interpolation_limit=interpolation_limit)
        array=self.interpolate(array=array,interpolation_method="bfill",interpolation_limit=interpolation_limit)
        if string_array:
            final_array=np.array([np.nan if np.isnan(vn) or (vn=="nan") else unmapping_dict[vn] for vn in array])
            self.array=final_array
        else:
            self.array=array


def stage_detector(y, lag=5, z_score=2,th_label=np.array([1,2]),influence=0):
    # assign stages based on th_label based on filtered y[i-lag:i] data.
    # step changed data is removed from the filtered Y

    th_label=np.array(th_label)
    filteredY = copy.deepcopy(y)  
    avgFilter=np.zeros_like(y)
    stdFilter=np.zeros_like(y)

    update_counter=0
    
    avgFilter = avgFilter+np.nanmean(y[0:lag+1]) #0,1,2,3,4
    stdFilter = stdFilter+np.nanstd(y[0:lag+1])
    current_label=np.nansum(th_label<np.nanmean(y[0:lag+1]))
    labels= np.zeros(len(y))+current_label
    new_label=[]
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > z_score * stdFilter [i-1]: # will skip in case nan
            # new signal
            labels[i]=current_label
            nlabel=np.sum(th_label<y[i])
            #print(f'i:{i},cnlabel:{nlabel}, new_label:{new_label},update_counter:{update_counter}')
            if (len(new_label)>0) and (nlabel!=new_label[-1]):
            
                # change in new label
                update_counter=1
                
                # keep current_label and start new_label counter again.
                filteredY[i-len(new_label):i]=avgFilter[i-len(new_label)-1]
                filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
                avgFilter[i] = np.nanmean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.nanstd(filteredY[(i-lag+1):i+1])
                new_label=[]
                new_label.append(nlabel)# assign pure signal
            else:
                new_label.append(nlabel)# assign pure signal
                    
                update_counter=update_counter+1
                
                filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
                avgFilter[i] = np.nanmean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.nanstd(filteredY[(i-lag+1):i+1])
                if update_counter>=lag:
                    # label change.
                    temp_vec=np.array(new_label)#labels[(i-lag+1):i+1]
                    temp_y=y[(i-lag+1):i+1]
                    vals, counts = np.unique(temp_vec, return_counts=True)
                    mode_value = vals[np.argwhere(counts == np.max(counts)).flatten()[0]]
                    current_label=mode_value
                    
                    avgFilter[i] = np.nanmean(temp_y[temp_vec==mode_value])
                    stdFilter[i] = np.nanstd(temp_y[temp_vec==mode_value])
                    temp_y[temp_vec!=mode_value]=avgFilter[i]

                    filteredY[(i-lag+1):i+1]=temp_y # update recent filter values.

                    labels[(i-lag+1):i+1]=mode_value # overwrite labels
                    update_counter=0
                    new_label=[]
        else:
            labels[i] = current_label
            if len(new_label)>0:
                # remove non-update values
                filteredY[i-len(new_label):i]=avgFilter[i-len(new_label)-1]
            new_label=[]
            filteredY[i] = y[i]
            avgFilter[i] = np.nanmean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.nanstd(filteredY[(i-lag+1):i+1])
            update_counter=0
            

    return dict(signals = np.asarray(labels),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter),
                filteredy=np.asarray(filteredY))





def make_monotonic(vec):
    for i,_ in enumerate(vec[:-1]):
        if np.isnan(vec[i+1]) or (vec[i+1]<vec[i]):
            vec[i+1]=vec[i]
    return vec


def find_mode(x):
    """
    This function finds the mode of a given list or numpy array.

    Parameters:
    x (list or np.ndarray): The input list or numpy array for which to find the mode.

    Returns:
    The mode of the input list or numpy array. If the input is a numpy array, it is first converted to a list.

    Raises:
    TypeError: If the input is neither a list nor a numpy array.
    """

    if type(x)==np.ndarray:
        x=x.tolist()
        
    occurence_count = Counter(x)
    return occurence_count.most_common(1)[0][0]



def ma_df(df,df_new,col,zid,Ts_smoothing,Ts_raw,method="mean"):
    # smoothing
    if zid is None:
        df_slice=copy.deepcopy(df[f'{col}'])
    else:
        df_slice=copy.deepcopy(df[f'{col}_{zid}'])
    if method=="mean":
        vec=df_slice.rolling(window=int(Ts_smoothing/Ts_raw),min_periods=1,center=True).mean().to_numpy() #proxy of Q
    
    elif method=="mode":
        vec=df_slice.rolling(window=int(Ts_smoothing/Ts_raw),min_periods=1,center=True).apply(lambda x: find_mode(x)).to_numpy() # mode
    else:
        raise ValueError("method should be either 'mean' or 'mode'.")

    if zid is None:
        df_new[f'{col}']=vec
    else:
        df_new[f'{col}_{zid}']=vec
    return df_new.copy()

def resample_df(raw_data,Ts,method="first"):
    
    if method=="mean":
        data=raw_data.resample(pd.Timedelta(f'{Ts}sec')).mean().copy()
    elif method=="first":
        data=raw_data.resample(pd.Timedelta(f'{Ts}sec')).first().copy()
    else:
        raise ValueError("method should be either 'mean' or 'first'.")

    return data


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    # are there ways doing plt in terminal?

    # test impute_change
    test_vec=np.array([np.nan,np.nan,np.nan,1.,2,3,4,5,6,10,12,13,14,15,16,17,np.nan,np.nan,30,31,35,32,33,34,40,45,46,47,48,49])
    imp=Imputation(test_vec,interpolation_method='nan',interpolation_limit=3)
    imp.impute_change(th=3)
    print(imp.array)


    # test stage detector 
    # TODO: does it accept NAN?
    # TODO: then, how to use the stage information for imputation?
    test_vec=np.array([0,0,0,0,1,1,1,0,1,0,1,1,1,0,0,0,0,0,0])
    stage_dict=stage_detector(y=test_vec, lag=3, z_score=2,th_label=np.array([0.5,1.5]),influence=0)
    print(stage_dict['signals'])

    # test median fileter (maybe not used..)
    test_vec=np.array([0.,1,2,3,4,5,6,7,100,-99,10,11,12,13,14,15,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0])
    imp=Imputation(test_vec,interpolation_method='nan',interpolation_limit=3)
    imp.impute_medianfilter(window=5,std_level=2,interpolation_method=None,interpolation_limit=None)
    print(imp.array)

    # test imupute state
    print('test_stage_detector')
    test_vec=np.array([0,0,0,0,1,1,1,0,1,0,1,1,1,0,0,0,0,0,np.nan,0,np.nan,0,1,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1]).astype('float')
    test_vec=np.array(['a','a','a','a','b','b','b','a','b','a','b',
                       'b','b','a','a','a','a','a',np.nan,'a',np.nan,'a',
                       'b','a','a','a','b','b','b','b','b','a','a','b','b','b','b','b'])
    imp=Imputation(test_vec,interpolation_method='nan',interpolation_limit=3)
    print(test_vec)
    imp.impute_state(n_state=2,state_th=0,interpolation_method='ffill',interpolation_limit=3)
    print(imp.array)
    
    
    # n_state=1,state_th=0,interpolation_method=None,interpolation_limit=None

    #impute_state
    #%%
    plt.plot(test_vec)
    # test state detector?


    ## should be done again.. checking non-numeric, null, etc.
    # def impute_type(self,data_type='float',method=None,limit=None):
    #     if method is None:
    #         method=self.method
    #     if limit is None:
    #         limit=self.limit
    #     array=copy.deepcopy(self.array)

    #     erroneous_indices = np.where(~np.isfinite(array))[0]
    #     array[erroneous_indices] = np.nan

    #     array=pd.Series(array).fillna(method=method).to_numpy()


    #     self.array=array


# df = pd.DataFrame({'Sales':[10,20,30,40,50,60,70], 'A':np.array([True,True,False,True,False,True,True]),
#                     'B':np.array(["o","o","x","o","x","o","o"])})
# df['rleid']=(df['A']!=df['A'].shift()).cumsum()




import numpy as np
import pandas as pd
import copy


class Imputation:
    def __init__(self,array,method='ffill',limit=3):
        self.array=array
        self.method=method
        self.limit=limit
        pass
    def interpolate(self,array,method,limit):
        if method =="nan":
            pass
        else:    
            array=pd.Series(array).interpolate(method=method,limit=limit).to_numpy()
        return array
    
    def impute_change(self, th=3,method=None,limit=None):
        if method == "NaN" or method == "nan":
            method="nan"
        elif method is None:
            method=self.method
        else:
            method=method

        if limit is None:
            limit=self.limit

        array=copy.deepcopy(self.array)
        prev_val=np.nan #array[0]
        for i in range(len(array)-1):
            if (np.abs(array[i+1]-array[i])>th) and np.isnan(prev_val):
                prev_val=array[i]
                array[i+1]=np.nan
            if np.isnan(array[i]) and not(np.isnan(array[i+1])):
                if np.abs(array[i+1]-prev_val)>th:
                    array[i+1]=np.nan


        differences = np.abs(np.diff(array))
        erroneous_indices = np.where(differences > th)[0] + 1
        array[erroneous_indices] = np.nan

        self.array=self.interpolate(array=array,method=method,limit=limit)
        

    def impute_minmax(self,min_val,max_val,method=None,limit=None):
        if method == "NaN" or method == "nan":
            method="nan"
        elif method is None:
            method=self.method
        else:
            method=method

        if limit is None:
            limit=self.limit
        array=copy.deepcopy(self.array)
        #array=self.interpolate(array=array,method=method,limit=limit)

        erroneous_indices = np.where((array < min_val) | (array > max_val))[0]
        array[erroneous_indices] = np.nan
        self.array=self.interpolate(array=array,method=method,limit=limit)

    def impute_zscore(self, zscore=3,method=None,limit=None):
        '''
        zscore: within the data, data outside of mean(+-)zscore*std will be removed.
        '''
        if method == "NaN" or method == "nan":
            method="nan"
        elif method is None:
            method=self.method
        else:
            method=method

        if limit is None:
            limit=self.limit
        array=copy.deepcopy(self.array)
        #array=self.interpolate(array=array,method=method,limit=limit)
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

        self.array=self.interpolate(array=array,method=method,limit=limit)
    
    def impute_state(self, n_state=1,state_th=0,method=None,limit=None):
        '''
        n_state: minimum number of state that is valid, i.e., state <n_state is replaced by nan
        '''
        if method == "NaN" or method == "nan":
            method="nan"
        elif method is None:
            method=self.method
        else:
            method=method

        if limit is None:
            limit=self.limit
        array=copy.deepcopy(self.array)

        array=self.interpolate(array=array,method=method,limit=limit)

        
        #rleid = np.cumsum(np.concatenate(([0], np.diff(array) != 0)))
        rleid = np.cumsum(np.concatenate(([0], np.abs(np.diff(array)) > state_th)))
    
        unique_rleid = np.unique(rleid)
        
        for rleid_val in unique_rleid:
            indices = np.where(rleid == rleid_val)[0]
            if len(indices) <= n_state:
                array[indices] = np.nan

        self.array=self.interpolate(array=array,method="ffill",limit=limit)
        self.array=self.interpolate(array=array,method="bfill",limit=limit)


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




# "R410a"

# import CoolProp.CoolProp as CP
# # http://www.coolprop.org/coolprop/HighLevelAPI.html#table-of-string-inputs-to-propssi-function
# # Using properties from CoolProp to get R410A density
# CP.PropsSI('H','T',300,'P',101325,'R410a') # J/kg
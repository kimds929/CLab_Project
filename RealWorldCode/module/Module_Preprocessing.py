import numpy as np
import pandas as pd

from datetime import datetime, timedelta
import holidays
kr_holidays = holidays.KR()

import torch


# # -------------------------------------------
# t = data['start_time'][0]
# t_next = data['start_time'][0] + timedelta(days=1)
# # t.year
# # t.weekday()
# # t in kr_holidays
# t.month
# 6 if t in kr_holidays else t.weekday()
# 6 if t_next in kr_holidays else t_next.weekday()
# # t.timestamp()
# # -------------------------------------------

def datetime_encoding(time_series):
    kr_holidays = holidays.KR()
    # time_series = data['start_time']
    month = time_series.apply(lambda x: np.nan if pd.isna(x) else x.month)
    weekday = time_series.apply(lambda x: np.nan if pd.isna(x) else (8 if x in kr_holidays else x.weekday()+1) )
    timestamp = time_series.apply(lambda x: np.nan if pd.isna(x) else (x.hour * 60 * 60 + x.minute  * 60 + x.second) )
    tomorrow_weekday = time_series.apply(lambda x: np.nan if pd.isna(x) else (8 if x+timedelta(days=1) in kr_holidays else (x+timedelta(days=1)).weekday()+1) )
    time_concat = pd.concat([month, weekday, timestamp, tomorrow_weekday], axis=1)
    time_concat.columns = [f"{time_series.name}_{n}"for n in ['month','weekday','timestamp', 't_weekwady']]
    return time_concat

# datetime_encoding(data['call_time_TimeMachine'])
# datetime_encoding(data['target_time'])

# lat_range = [33, 43]
# lng_range = [124, 132] 
    



##############################################################################################################################
class BaroNowPreprocessing:
    def __init__(self, data=None, feature_dict=None):
        self.transportation_code = {'car_tmap':0, 'car_kakao': 1}

        self.data = data
        self.feature_dict = feature_dict
        
        self.type_transform_df = None
        self.encoding_df = None
        self.encoding_dict = {}
        self.encoding_cols_dict = {}
        self.torch_data = None
    
    # (type_transform)
    def type_transform(self, data=None):
        df_contexts = data.copy() if data is not None else self.data.copy()
        target_time_safe_margin = 8*60

        float_cols = ['round', 'path_time_TimeMachine', 'path_time_LastAPI']
        time_cols = ['start_time', 'target_time', 'call_time_TimeMachine', 'cur_time','call_time_LastAPI']

        # for lc in loc_cols:
        #     df_contexts[lc] = df_contexts[lc].apply(lambda x: np.nan if (type(x)==float and pd.isna(x)) else (x if (type(x) == list or type(x) == tuple) else eval(x)))

        for fc in float_cols:
            if fc in df_contexts.columns:
                df_contexts[fc] = df_contexts[fc].astype(float)

        for tc in time_cols:
            if tc in df_contexts.columns:
                if 'datetime' not in str(df_contexts[tc].dtype):
                    df_contexts[tc] = df_contexts[tc].apply(lambda x: np.nan if pd.isna(x) else datetime.strptime(x, "%Y-%m-%dT%H:%M:%S%z"))

        # TimeMachine 기준 출발시간 계산
        # df_contexts['call_time_TimeMachine'] = df_contexts.apply(lambda x: x['target_time'] - timedelta(seconds=x['path_time_TimeMachine']), axis=1)
        # df_contexts['req_leaving_time_TimeMachine'] = df_contexts.apply(lambda x: x['target_time'] - timedelta(seconds=x['path_time_TimeMachine']) - timedelta(seconds=target_time_safe_margin), axis=1)

        # TimeMachine 기준 출발시간과 현재시간 차이 계산 (second 단위)
        # df_contexts['req_leaving_time_delta_TimeMachine'] = df_contexts.apply(lambda x: (x['req_leaving_time_TimeMachine'] - x['cur_time']).total_seconds(), axis=1)
        self.type_transform_df = df_contexts
        return df_contexts
    
    # (encoding)
    def encoding(self, data=None, feature_dict=None, fillna=None):
        if data is None:
            if self.type_transform_df is not None:
                df_contexts = self.type_transform_df.copy()
            else:
                raise Exception('Not preprocessing yet.')
        else:
            df_contexts = data
        feature_dict = feature_dict if feature_dict is not None else self.feature_dict

        # token_df
        token_df = pd.DataFrame()
        if 'token_cols' in feature_dict.keys():
            if len(feature_dict['token_cols']) > 0:
                token_df = df_contexts[feature_dict['token_cols'][0]].apply(lambda x: self.transportation_code[x]).to_frame()

        # numerical_columns
        numerical_df = pd.DataFrame()
        if 'numerical_cols' in feature_dict.keys():
            numerical_cols = feature_dict['numerical_cols']
            if len(numerical_cols) > 0:
                numerical_df = df_contexts[numerical_cols]
                numerical_df.index = df_contexts.index

        # temporal df
        temporal_df = pd.DataFrame()
        if 'temporal_cols' in feature_dict.keys():
            temporal_cols = feature_dict['temporal_cols']
            if len(temporal_cols) > 0:
                for tc in temporal_cols:
                    # if 'float' in str(df_contexts[tc].dtype) or 'int' in str(df_contexts[tc].dtype):
                    #     temporal_df = pd.concat([temporal_df, df_contexts[tc]], axis=1)
                    # else:
                    temporal_df = pd.concat([temporal_df, datetime_encoding(df_contexts[tc])], axis=1)
                temporal_df.index = df_contexts.index

        # spatial_columns
        spatial_scales={'lat_range':[33,43], 'lng_range':[124,132]}
        spatial_df = pd.DataFrame()
        if 'spatial_cols' in feature_dict.keys():
            spatial_cols = feature_dict['spatial_cols']
            if len(spatial_cols) > 0:
                spatial_cols_transform = list(np.stack([[f"{cols}_x", f"{cols}_y"] for cols in spatial_cols]).ravel())
                spatial_arr_stack = np.stack(list(df_contexts[spatial_cols].applymap(lambda x: np.array(eval(x)[:2])).to_dict('list').values())).astype(np.float32)
                spatial_arr = spatial_arr_stack.transpose(1,0,2).reshape(-1,4)
                spatial_df = pd.DataFrame(spatial_arr, columns=spatial_cols_transform)
                if spatial_scales is not None:
                    gps_ranges = np.tile(np.stack(list(np.array(list(spatial_scales.values())).T)), 2)
                    spatial_df = (spatial_df - gps_ranges[0]) / (gps_ranges[1] - gps_ranges[0])
                spatial_df.index = df_contexts.index

        # # concat
        self.encoding_dict['token'] = token_df
        self.encoding_dict['numerical'] = numerical_df
        self.encoding_dict['temporal'] = temporal_df
        self.encoding_dict['spatial'] = spatial_df
        
        self.encoding_df = pd.concat([token_df, temporal_df, spatial_df, numerical_df], axis=1)

        if fillna is not None:
            self.encoding_df = self.encoding_df.fillna(fillna)
            self.encoding_dict['token'] = token_df if fillna is None else token_df.fillna(fillna)
            self.encoding_dict['numerical'] = numerical_df if fillna is None else numerical_df.fillna(fillna)
            self.encoding_dict['temporal'] = temporal_df if fillna is None else temporal_df.fillna(fillna)
            self.encoding_dict['spatial'] = spatial_df if fillna is None else spatial_df.fillna(fillna)
            
        self.encoding_cols_dict['token'] = list(token_df.columns)
        self.encoding_cols_dict['numerical'] = list(numerical_df.columns)
        self.encoding_cols_dict['temporal'] = list(temporal_df.columns)
        self.encoding_cols_dict['spatial'] = list(spatial_df.columns)

        return (self.encoding_df, self.encoding_cols_dict)

    # (to_torch)
    def to_torch(self, encoding_df=None):
        if encoding_df is None:
            if self.encoding_df is not None:
                encoding_df = self.encoding_df.copy()
            else:
                raise Exception('Not encoding yet.')
        else:
            encoding_df = encoding_df
        
        self.torch_data = torch.tensor(np.array(encoding_df)).type(torch.float32)
        return self.torch_data
    
    # (preprocessing)
    def preprocessing(self, data=None, feature_dict=None, fillna=None):
        data = data.copy() if data is not None else self.data.copy()
        self.type_transform(data)
        self.encoding(feature_dict=feature_dict, fillna=fillna)
        return self.to_torch()

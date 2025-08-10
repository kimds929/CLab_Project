############################################################################################################
# (Path Setting)
import os
import sys
file_name = os.path.abspath(__file__)
file_path = os.path.dirname(file_name)
base_path = '/'.join(file_path.split('/')[:[i for i, d in enumerate(file_path.split('/')) if 'BaroNowProject' in d][0]+1])
sys.path.append( base_path )
os.chdir(base_path)
print(f"(file) {file_name.split('/')[-1]}, (base_path) {base_path}")

cold_start_path = os.path.join(base_path, 'cold_start');  sys.path.append( cold_start_path )
dataset_path = os.path.join(base_path, 'dataset');  sys.path.append( dataset_path )
env_path = os.path.join(base_path, 'environment');  sys.path.append( env_path )
logging_path = os.path.join(base_path, 'logging');  sys.path.append( logging_path )
module_path = os.path.join(base_path, 'module');    sys.path.append( module_path )
model_path = os.path.join(base_path, 'model');  sys.path.append( model_path )
weight_path = os.path.join(base_path, 'weight');    sys.path.append( weight_path )

crawlling_path = f"{os.path.dirname(base_path)}/baroNow_data/"
crawlling_car_path = f"{crawlling_path}/data_car"
############################################################################################################

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datetime import datetime, timedelta
import pytz
from six.moves import cPickle
##############################################################################################################


kst = pytz.timezone('Asia/Seoul')
now_date = datetime.strftime(datetime.now(kst), '%Y%m%d')

save_file_name = f'Alg4_TrainingData'

# load json ----------------------------------------------------------------
if f"{save_file_name}.json" in os.listdir(dataset_path):
    with open(f"{dataset_path}/{save_file_name}.json", 'r', encoding='utf-8') as file:
        meta_data = json.load(file)
else:
    meta_data = {
            "header": {
                "title": None,
                "description":None,
                "last_update": datetime.strftime(datetime.now(kst), "%Y-%m-%dT%H:%M:%S%z") ,
                "path": None
            },
            "body": {
                "contents": []
            }
        }
    meta_data['header']['title'] = f'{save_file_name}'
    meta_data['header']['description'] = "Alg4 Training을 위한 Dataset"
# ---------------------------------------------------------------------------


#############################################################################
feature_dict = {}
feature_dict['token_cols'] = ['transportation']
feature_dict['temporal_cols'] = ['target_time', 'call_time_LastAPI']
feature_dict['spatial_cols'] = ['start_point', 'target_point']
feature_dict['numerical_cols'] =  ['path_time_TimeMachine','path_time_LastAPI']

from functools import reduce
total_cols = reduce( lambda x,y: x+y , list(feature_dict.values()) )

################################################################################

df_optimal = pd.read_csv(f"{dataset_path}/Optimal_API_Call_DataSet.csv", encoding='utf-8-sig')
print(df_optimal.shape)

df_optimal_firstline = df_optimal.groupby('id').head(1)
df_optimal_firstline['optimal_delta'] = df_optimal_firstline.apply(lambda x: x['path_time_LastAPI'] - x['path_time_TimeMachine'], axis=1)



################################################################################
file_names = [d for d in os.listdir(crawlling_car_path) if ('csv' in d and d[:2] == '20')]
file_names = [fn for fn in file_names if now_date not in fn]    # 오늘 제외 (data 쌓고있는 중)
# file_names = [fn for fn in file_names if fn not in meta_data['body']['contents']]    # 이미 저장되어있는 날짜 제외
file_names = sorted(file_names)
np.array(file_names)

data_timemachine = pd.DataFrame()
data_realtime = pd.DataFrame()
saved_file_names = []


for file_name in tqdm(file_names):
    print(file_name)
    data = pd.read_csv(f"{crawlling_car_path}/{file_name}", encoding='utf-8-sig', sep=",")

    # time_machine ----------------------------------------------------------------------------------------------------------------------------
    df_timemachine = data[(data['group'] == 'timemachine') ][['id','group']+total_cols]

    # merge
    df_timemachine_merge = pd.merge(left=df_timemachine, right=df_optimal_firstline[['id','optimal_delta']], on='id', how='left')
    df_timemachine_merge = df_timemachine_merge[~df_timemachine_merge['optimal_delta'].isna()]
    data_timemachine_segment = df_timemachine_merge.reset_index(drop=True)

    

    # real_time ----------------------------------------------------------------------------------------------------------------------------
    df_realtime = data[data['group'] == 'realtime'][['id','group']+total_cols]

    # merge
    df_realtime_merge = pd.merge(left=df_realtime, right=df_optimal_firstline[['id','optimal_delta']], on='id', how='left')
    df_realtime_merge = df_realtime_merge[~df_realtime_merge['optimal_delta'].isna()]
    data_realtime_segment = df_realtime_merge.reset_index(drop=True)

    # concat
    data_timemachine = pd.concat([data_timemachine, data_timemachine_segment], axis=0).reset_index(drop=True)
    data_realtime = pd.concat([data_realtime, data_realtime_segment], axis=0).reset_index(drop=True)


    # save
    cPickle.dump(data_timemachine, open(f"{dataset_path}/{save_file_name}_timemachine.pkl", 'wb'))
    cPickle.dump(data_realtime, open(f"{dataset_path}/{save_file_name}_realtime.pkl", 'wb'))
    data_timemachine.to_csv(f"{dataset_path}/{save_file_name}_timemachine.csv", index=False, encoding='utf-8-sig')
    data_realtime.to_csv(f"{dataset_path}/{save_file_name}_realtime.csv", index=False, encoding='utf-8-sig')

    # add path info
    meta_data['header']['last_update'] = datetime.strftime(datetime.now(kst), "%Y-%m-%dT%H:%M:%S%z")
    meta_data['header']['path'] = [f"{dataset_path}/{save_file_name}_timemachine.csv", f"{dataset_path}/{save_file_name}_timemachine.pkl", f"{dataset_path}/{save_file_name}_realtime.csv", f"{dataset_path}/{save_file_name}_realtime.pkl"]

    saved_file_names += [file_name]
    meta_data['body']['contents'] = sorted(saved_file_names)
    # save info to json
    with open(f"{dataset_path}/{save_file_name}.json", 'w', encoding='utf-8') as file:
        json.dump(meta_data, file, indent=4, ensure_ascii=False) 
print('concat complete.')

save_file_name = f'Optimal_API_Call_DataSet'
print(save_file_name)

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



import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datetime import datetime, timedelta
import pytz

import json
from six.moves import cPickle

##############################################################################################################
kst = pytz.timezone('Asia/Seoul')
now_date = datetime.strftime(datetime.now(kst), '%Y%m%d')
target_time_safe_margin = 8*60      # # safe margin

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
    meta_data['header']['description'] = "Target_time - 8분에 도착할 수 있는 optimal time step에 API Call을 한 Event"
# ---------------------------------------------------------------------------

float_cols = ['round', 'path_time_TimeMachine', 'path_time_LastAPI', 'path_time_ans']
time_cols = ['start_time', 'target_time', 'call_time_TimeMachine', 'cur_time','call_time_LastAPI', 'req_leaving_time']

file_names = [d for d in os.listdir(crawlling_car_path) if ('csv' in d and d[:2] == '20')]
file_names = [fn for fn in file_names if now_date not in fn]    # 오늘 제외 (data 쌓고있는 중)
file_names = [fn for fn in file_names if fn not in meta_data['body']['contents']]    # 이미 저장되어있는 날짜 제외
file_names = sorted(file_names)


# file_name = '20240922.csv'
data_update_optimal_api_call = pd.DataFrame()
for file_name in tqdm(file_names):
    print(file_name)
    data = pd.read_csv(f"{crawlling_car_path}/{file_name}", encoding='utf-8-sig', sep=",")
    data = data[data['group'].isin(['timemachine', 'realtime'])]

    for fc in float_cols:
        data[fc] = data[fc].astype(float)

    for tc in time_cols:
        data[tc] = data[tc].apply(lambda x: np.nan if pd.isna(x) else datetime.strptime(x, "%Y-%m-%dT%H:%M:%S%z"))

    data_time_machine = data[data['group'] == 'timemachine']
    data_real_time = data[data['group'] == 'realtime']

    if len(data_real_time) > 0:
        # TimeMachine 기준 출발시간 계산
        data_real_time['req_leaving_time_TimeMachine'] = data_real_time.apply(lambda x: x['target_time'] - timedelta(seconds=x['path_time_TimeMachine']) - timedelta(seconds=target_time_safe_margin), axis=1)

        # TimeMachine 기준 출발시간과 현재시간 차이 계산 (second 단위)
        data_real_time['req_leaving_time_delta_TimeMachine'] = data_real_time.apply(lambda x: (x['req_leaving_time_TimeMachine'] - x['cur_time']).total_seconds(), axis=1)

        # Current Time + path_time_LastAPI 가 target_time - 8분과 120초 이내인 event만 filtering
        data_real_time['diff_from_optimal_api_call'] = data_real_time.apply(lambda x: ( x['target_time'] - timedelta(seconds=8*60) - (x['cur_time'] + timedelta(seconds=x['path_time_LastAPI'])) ).total_seconds(), axis=1)
        # plt.hist(data_real_time['diff_from_optimal_api_call'],bins=30)
        data_real_time_optimal_candidates = data_real_time[data_real_time['diff_from_optimal_api_call'].apply(abs)<120]

        # ID별로 Current Time + path_time_LastAPI 가 target_time - 8분과 120초 이내인 even중에 가장 가까운 data만 filtering
        data_real_time_optimal = data_real_time_optimal_candidates.groupby('id').apply(lambda x: x[x['diff_from_optimal_api_call'].abs() == x['diff_from_optimal_api_call'].abs().min()])
        data_real_time_optimal = data_real_time_optimal.reset_index('id', drop=True)
        data_real_time_optimal = data_real_time_optimal.groupby('id').apply(lambda x: x[x['diff_from_optimal_api_call'] == x['diff_from_optimal_api_call'].max()])
        data_real_time_optimal = data_real_time_optimal.reset_index('id', drop=True)
        
        for tc in time_cols + ['req_leaving_time_TimeMachine']:
            data_real_time_optimal[tc] = data_real_time_optimal[tc].apply(lambda x: np.nan if pd.isna(x) else datetime.strftime(x, "%Y-%m-%dT%H:%M:%S%z"))

        data_real_time_optimal.insert(0, 'file_name', int(file_name.replace('.csv','')))

        # # (save DataFrame)
        # cPickle.dump(data_real_time_nearest_TM, open(f"{dataset_path}/{file_name.replace('.csv','')}.pkl", 'wb'))
        # print(file_name.replace(".csv",""), "save complete.")

        data_update_optimal_api_call = pd.concat([data_update_optimal_api_call, data_real_time_optimal], axis=0)
        meta_data['body']['contents'].append(file_name)
# list(data_update_optimal_api_call['file_name'].value_counts().sort_index().index)



# save_data -----------------------------------------------------------------------------------------

# load data
if f'{save_file_name}.pkl' in os.listdir(dataset_path):        # pkl로 불러오기
    data_optimal_api_call = cPickle.load(open(f"{dataset_path}/{save_file_name}.pkl", 'rb'))
elif f'{save_file_name}.csv' in os.listdir(dataset_path):      # csv로 불러오기
    data_optimal_api_call = pd.read_csv(f"{dataset_path}//{save_file_name}.csv", encoding="utf-8-sig")
else:
    data_optimal_api_call = pd.DataFrame()

# concat
data_optimal_api_call = pd.concat([data_optimal_api_call, data_update_optimal_api_call], axis=0)

# save
cPickle.dump(data_optimal_api_call, open(f"{dataset_path}/{save_file_name}.pkl", 'wb'))
data_optimal_api_call.to_csv(f"{dataset_path}/{save_file_name}.csv", index=False, encoding='utf-8-sig')

# add path info
meta_data['header']['last_update'] = datetime.strftime(datetime.now(kst), "%Y-%m-%dT%H:%M:%S%z")
meta_data['header']['path'] = [f"{dataset_path}/{save_file_name}.csv", f"{dataset_path}/{save_file_name}.pkl"]
meta_data['body']['contents'] = sorted(meta_data['body']['contents'])
# save info to json
with open(f"{dataset_path}/{save_file_name}.json", 'w', encoding='utf-8') as file:
    json.dump(meta_data, file, indent=4, ensure_ascii=False) 
print('concat complete.')
# ---------------------------------------------------------------------------------------------------

# Performance -------------------------------------------------------------
plt.hist(data_optimal_api_call['diff_from_optimal_api_call']/60, bins=30, color='skyblue', edgecolor='gray')
plt.show()
# --------------------------------------------------------------------


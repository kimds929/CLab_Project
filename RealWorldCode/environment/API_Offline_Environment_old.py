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
############################################################################################################

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

import json
import subprocess
from six.moves import cPickle

# BaroNow Prediction Class Load
from Run_Prediction import BaroNowPrediction as Prediction
# from Run_RuleBased import BaroNowRuleBased as Prediction

######################################################################

######################################################################
# 쌓여진 특정 OfflineData에서 Random하게 1개의 Episode를 추출해서 실행
def importenv_from_offlinedata(file_name='20241001', id=None, dynamic=False, **kwargs):
    load_data = pd.read_csv(f"{base_path}/dataset/{file_name}.csv", encoding='utf-8-sig')
    load_data = load_data[load_data['group'] == 'realtime']
    if id is not None:
        selected_id = id
    else:
        selected_id = load_data['id'].drop_duplicates().sample().item() # random_id select
    data_api = load_data[load_data['id'] == selected_id]
    
    if dynamic:
        x_init, y_init, _ = eval(data_api.iloc[0]['start_point'])
        x_scale, y_scale = np.random.normal(loc=0.025, scale=0.01), np.random.normal(loc=0.02, scale=0.01)
        x_points = [x_init] + list( x_init + x_scale * np.random.normal(loc=0, scale=1, size=len(data_api['start_point'])-1) )
        y_points = [y_init] + list( y_init + y_scale * np.random.normal(loc=0, scale=1, size=len(data_api['start_point'])-1) )

        # data_api['cur_point'] = pd.Series([(xp, yp, "-") for xp, yp in zip(x_points, y_points)], index=data_api.index).astype(str)
        data_api['cur_point'] = pd.Series([(xp, yp) for xp, yp in zip(x_points, y_points)], index=data_api.index).astype(str)
    return data_api


################################################################################
# data_api = importenv_from_offlinedata()
data_api = importenv_from_offlinedata(dynamic=True)


# data_api = importenv_from_offlinedata(id='GS25_산성역자이점_주차장_2024-10-01T11:47:38+0900')
# data_api = importenv_from_offlinedata(id='GS25_산성역자이점_주차장_2024-10-01T11:47:38+0900', dynamic=True)
# data_api = importenv_from_offlinedata(id ='CU_길음역점_주차장_2024-10-01T21:44:45+0900', dynamic=True)
# data_api = importenv_from_offlinedata(id = 'GS25_용인스마트점_2024-10-01T13:00:20+0900')

baronow_prediction = Prediction()
################################################################################

# data_api['cur_point'].iloc[0]
#------------------------------------------------------------------------------------
data_api['transportation'] = 'car_kakao'
# data_api['transportation'] = 'car_tmap'
print(data_api.iloc[0]['transportation'])
#------------------------------------------------------------------------------------


output = {}
output['id'] = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')
output['contexts'] = {}
output['status'] = 0        # 0: 진행, 1: 출발에 의한 종료, 2: New Episode에 의한 종료
output['descriptions'] = None

round = 0
cur_time = datetime.strptime(data_api.iloc[0]['cur_time'], "%Y-%m-%dT%H:%M:%S%z")
api_call_time = None
leaving_time = None
data_history = pd.DataFrame()
new_episodes = None


n_iter = len(data_api)
for _ in range(n_iter):
    # ---------------------------------------------------------------------------------------
    # (API Call 시간 도래시)
    if (api_call_time is not None) and (cur_time > datetime.strptime(api_call_time, "%Y-%m-%dT%H:%M:%S%z")):
        context = data_api.iloc[[round], :]     # context with API Call
        api_call_time = None    # API Call정보 초기화
        data_history = pd.concat([data_history, context], axis=0)
    
    # API Call 시간을 argument로 받지 않은 경우 or API시간이 아직 다가오지 않은 경우
    else:   
        data_noapi = data_api.iloc[[round], :]
        for api_col in ['call_time_LastAPI', 'call_point_LastAPI', 'path_LastAPI', 'path_time_LastAPI']:
            if len(data_history) > 0:
                data_noapi[api_col] = data_history.iloc[-1,:][api_col]
            else:
                data_noapi[api_col] = np.nan
        context = data_noapi.copy()
        data_history = pd.concat([data_history, context], axis=0)  

        # ---------------------------------------------------------------------------------------
        # (NewEpisode Time 도래시)
        if (new_episodes is not None) and (
             (leaving_time is None and cur_time > datetime.strptime(new_episodes, "%Y-%m-%dT%H:%M:%S%z"))
             or (leaving_time is not None and datetime.strptime(leaving_time,"%Y-%m-%dT%H:%M:%S%z") > datetime.strptime(new_episodes,"%Y-%m-%dT%H:%M:%S%z")) 
             ):
                output['status'] = 2     # 종료조건 인자
                output['descriptions'] = "New_Episodes"
        # ---------------------------------------------------------------------------------------
        
        # ---------------------------------------------------------------------------------------
        # (Leaving Time 도래시)
        elif (leaving_time is not None) and (cur_time > datetime.strptime(leaving_time, "%Y-%m-%dT%H:%M:%S%z")):
            context = data_api.iloc[[round], :]     # context with API Call
            data_history = pd.concat([data_history, context], axis=0)
            
            output['status'] = 1     # 종료조건 인자
            output['descriptions'] = "Leaving to Destination. End_of_episodes."
        # ---------------------------------------------------------------------------------------      


    # ---------------------------------------------------------------------------------------
    # contexts 전달시 dictionary(Json) 형태로 변환
    
    output['contexts'] = data_history.applymap(lambda x: None if pd.isna(x) else x).to_dict('records')
    # print(output['contexts'])
    # Run_Prediction.py에 context전달
    # response_pred = baronow_prediction.predict(output, mode='offline')
    response_pred = baronow_prediction.predict(output, mode='offline', return_full_info=True)
    # response_pred = baronow_prediction.predict(output, mode='online', return_full_info=True)
    
    # ---------------------------------------------------------------------------------------
    if ('api_call_time' in response_pred.keys()) and (response_pred['api_call_time'] is not None):
        api_call_time = response_pred['api_call_time']

    # 출발시간 정보
    if ('leaving_time' in response_pred.keys()) and (response_pred['leaving_time'] is not None):
        leaving_time = response_pred['leaving_time']
    
    # 위치이동에 따른 기존 Episode 종료 및 새로운 Episode시작
    if ('new_episodes' in response_pred.keys()) and (response_pred['new_episodes'] is not None):
        new_episodes = response_pred['new_episodes']
    # ---------------------------------------------------------------------------------------

    # print log
    print(_, cur_time, output['descriptions'], response_pred)

    # ---------------------------------------------------------------------------------------
    # Episode 종료
    if output['status'] > 0:
        break

    # time step / round 증가 모사
    round += 1
    cur_time = cur_time + timedelta(minutes=2)
    # ---------------------------------------------------------------------------------------


# 종료조건
if output['status'] == 1:
    # Performance Evaluation ----------------------------------------------------------------------------------------
    last_context = pd.DataFrame(data_history).iloc[-1,:]
    if not pd.isna(last_context['path_time_LastAPI']):
        arrival_time = datetime.strptime(last_context['cur_time'], "%Y-%m-%dT%H:%M:%S%z") + timedelta(seconds=last_context['path_time_LastAPI'])
        residual_arrival_time = (arrival_time - datetime.strptime(last_context['target_time'], "%Y-%m-%dT%H:%M:%S%z") ).total_seconds()/60
        print(f"Is in allowance range? : {-13 < residual_arrival_time < 0}, {residual_arrival_time}")


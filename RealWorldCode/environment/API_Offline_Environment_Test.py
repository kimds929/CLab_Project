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

from tqdm.auto import tqdm
import pytz
import json
import subprocess
from six.moves import cPickle

# BaroNow Prediction Class Load
from Run_Prediction import BaroNowPrediction as Prediction
# from Run_RuleBased import BaroNowRuleBased as Prediction


n_iter = 10000


test_case = 2       # 1
model_name = 'transformer'  # 2
alpha = 4   # 3
gamma = 2.5 # 4
max_api_call = None # 5
min_api_interval = 3    # 6

kst = pytz.timezone('Asia/Seoul')
now_date = datetime.strftime(datetime.now(kst), "%Y-%m-%dT%H:%M:%S%z")



print()
print( "*** initial setting *** ")
if ('ipykernel_launcher' not in sys.argv[0]) and  len(sys.argv) > 0:
    if len(sys.argv) > 1:
        test_case = int(sys.argv[1])

    if len(sys.argv) > 2:
        model_name = sys.argv[2]
        
    if len(sys.argv) > 3:
    #     n_iter = int(sys.argv[2])
        alpha = float(sys.argv[3])
    
    if len(sys.argv) > 4:
        gamma = float(sys.argv[4])

    if len(sys.argv) > 5:
        if sys.argv[5] == 'None':
            max_api_call = None
        else:
            max_api_call = int(sys.argv[5])

    if len(sys.argv) > 6:
        min_api_interval = int(sys.argv[6])

test_case_name = None
test_case_group = None
if test_case == 1:
    test_case_name = '_20241020-20241030'
    test_case_group = 'all'
elif test_case == 2:
    test_case_name = '_20241207-20241217'
    test_case_group = 'commute'
elif test_case == 3:
    test_case_name = '_20241020-20241217'
    test_case_group = 'combine'

print(f'(input_arguments) test_case: {test_case_group} / model: {model_name} / alpha: {alpha:.1f} / gamma: {gamma: .1f} / max_api_call: {max_api_call} / min_api_interval: {min_api_interval}')
    # print(f'(input_arguments) model: {model_name} / n_iter: {n_iter} / alpha : {alpha:.1f} / gamma : {gamma: .1f}')


save_file_name_base = f'{test_case_group}_{now_date}_{model_name}_alpha{alpha}_gamma{gamma}_max_api_call{max_api_call}_min_api_interval{min_api_interval}_offline_test_result'
print(save_file_name_base)


baronow_prediction = Prediction(model=model_name, alpha=alpha, gamma=gamma, max_api_call=max_api_call, min_api_interval=min_api_interval)


######################################################################
test_mode = 'offline'
transportation = np.random.choice(['car_kakao', 'car_tmap'])
print(f"test_mode : {test_mode}")

############################################################################################################



######################################################################
def load_dataset(file_name='20241001', folder_path=None):
    data_path = dataset_path
    if folder_path is not None:
        data_path = folder_path
    load_data = pd.read_csv(f"{data_path}/{file_name}.csv", encoding='utf-8-sig')
    load_data = load_data[load_data['group'] == 'realtime']
    return load_data

# 쌓여진 특정 OfflineData에서 Random하게 1개의 Episode를 추출해서 실행
def importenv_from_offlinedata(load_data, id=None, dynamic=False, **kwargs):
    if id is not None:
        selected_id = id
    else:
        selected_id = load_data['id'].drop_duplicates().sample().item() # random_id select
    data_api = load_data[load_data['id'] == selected_id]
    
    if dynamic:
        x_init, y_init, r_idx = eval(data_api.iloc[0]['start_point'])
        x_scale, y_scale = np.random.normal(loc=0.025, scale=0.01), np.random.normal(loc=0.02, scale=0.01)
        x_points = [x_init] + list( x_init + x_scale * np.random.normal(loc=0, scale=1, size=len(data_api['start_point'])-1) )
        y_points = [y_init] + list( y_init + y_scale * np.random.normal(loc=0, scale=1, size=len(data_api['start_point'])-1) )

        # data_api['cur_point'] = pd.Series([(xp, yp, "-") for xp, yp in zip(x_points, y_points)], index=data_api.index).astype(str)
        data_api['cur_point'] = pd.Series([(xp, yp) for xp, yp in zip(x_points, y_points)], index=data_api.index).astype(str)
    return data_api


################################################################################

folder_path = f"{os.path.dirname(base_path)}/baroNow_data/data_car"

# file_list = []
# for i in range(10):
#     file_list.append(i + 20241020)
# np.array(file_list)


# data_all = pd.DataFrame()
# file_names1 = [20241020, 20241021, 20241023, 20241024, 20241025, 20241026, 20241027, 20241028, 20241029, 20241030]
# file_names2 = [20241208, 20241209, 20241210, 20241211, 20241212, 20241213, 20241214, 20241215, 20241216, 20241217]
# file_names = file_names1 + file_names2
# for f in tqdm(file_names):
#     data_temp = load_dataset(folder_path, str(f))
#     data_all = pd.concat([data_all, data_temp], axis=0)
# data_all = data_all.reset_index(drop=True)
# cPickle.dump(data_all, open(f"{folder_path}/_{file_names[0]}-{file_names[-1]}.pkl", 'wb'))
# print('save.')


# dataset = load_dataset('20241213', folder_path)
dataset = cPickle.load(open(f"{folder_path}/{test_case_name}.pkl", 'rb'))







# verbose = 1     # display
verbose = 0     # display

# n_iter = 100
info_list = []
for _ in tqdm(range(n_iter)):
    info_dict = {}

    # ----------------------------
    data_api = importenv_from_offlinedata(dataset)
    # data_api = importenv_from_offlinedata(dataset, dynamic=True)
    # data_api = importenv_from_offlinedata(dataset)
    data_api = data_api.drop(['req_leaving_time', 'path_time_ans'], axis=1)
    data_api = data_api[data_api['transportation'] == 'car_kakao']

    arrival = data_api.apply(lambda x: (datetime.strptime(x['target_time'],"%Y-%m-%dT%H:%M:%S%z")
                - datetime.strptime(x['cur_time'],"%Y-%m-%dT%H:%M:%S%z") -timedelta(seconds=x['path_time_LastAPI']) ).total_seconds()/60
                , axis=1)

    output = {}
    output['id'] = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')
    output['contexts'] = []
    output['status'] = 0        # 0: 진행, 1: 출발에 의한 종료, 2: New Episode에 의한 종료
    output['descriptions'] = None


    round = 0
    n_api_call = 0
    cur_time = datetime.strptime(data_api.iloc[0]['cur_time'], "%Y-%m-%dT%H:%M:%S%z")
    api_call_time = None
    leaving_time = None
    data_history = pd.DataFrame()
    new_episodes = None

    n_iter = len(data_api)
    for r_idx in range(n_iter):
        if arrival.iloc[1] < 0:
            break
        # ---------------------------------------------------------------------------------------
        # (API Call 시간 도래시)
        if (api_call_time is not None) and (cur_time > datetime.strptime(api_call_time, "%Y-%m-%dT%H:%M:%S%z")):
            context = data_api.iloc[[round], :]     # context with API Call
            api_call_time = None    # API Call정보 초기화
            data_history = pd.concat([data_history, context], axis=0)
            n_api_call += 1     # api call 횟수 추가
        
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
        
        response_pred = baronow_prediction.predict(output, mode=test_mode, return_full_info=True)
        # response_pred = baronow_prediction.predict(output, mode='offline', return_full_info=True)
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
        if verbose > 0:
            print(r_idx, cur_time, output['descriptions'], response_pred)

        # ---------------------------------------------------------------------------------------
        # Episode 종료
        if output['status'] > 0:
            break

        # time step / round 증가 모사
        round += 1
        if r_idx + 1 < len(data_api):
            cur_time = datetime.strptime(data_api.iloc[r_idx+1]['cur_time'], "%Y-%m-%dT%H:%M:%S%z")
            # cur_time = cur_time + timedelta(minutes=2)
        else:
            break
        # ---------------------------------------------------------------------------------------


    # 종료조건
    if arrival.iloc[1] < 0:
        print('not valid data')
    elif output['status'] == 1:
        # Performance Evaluation ----------------------------------------------------------------------------------------

        last_context = pd.DataFrame(data_history).iloc[-1,:]
        if not pd.isna(last_context['path_time_LastAPI']):
            arrival_time = datetime.strptime(last_context['cur_time'], "%Y-%m-%dT%H:%M:%S%z") + timedelta(seconds=last_context['path_time_LastAPI'])
            residual_arrival_time = (arrival_time - datetime.strptime(last_context['target_time'], "%Y-%m-%dT%H:%M:%S%z") ).total_seconds()/60
            arrival_TF = -13 < residual_arrival_time < 0

            print(f"\t n_api_call : {n_api_call} / n_of_OfflineData : {r_idx}")
            print(f"\t Is in allowance range? : {arrival_TF}, {residual_arrival_time}")
        

            # save info
            info_dict['id'] = data_api.iloc[0]['id']
            info_dict['arrival'] = arrival_TF
            info_dict['result'] = residual_arrival_time
            info_dict['n_of_api_call'] = n_api_call
            info_dict['n_of_OfflineData'] = r_idx
            info_list.append(info_dict)
    
    elif output['status'] == 2:
        print("\t New Episode End.")
    
    else:
        print(f"\t n_api_call : {n_api_call} / n_of_OfflineData : {r_idx}")
        print("\t False")
        # save info
        info_dict['id'] = data_api.iloc[0]['id']
        info_dict['arrival'] = False
        info_dict['result'] = None
        info_dict['n_of_api_call'] = n_api_call
        info_dict['n_of_OfflineData'] = r_idx
        info_list.append(info_dict)
    
    # save result
    pd.DataFrame(info_list).to_csv(f"{env_path}/offline_test_result/{save_file_name_base}.csv", index=False, encoding='utf-8-sig')


df_result = pd.DataFrame(info_list)
n_episodes = len(df_result)
accept_ratio = df_result['arrival'].sum() / len(df_result)
excess_ratio = df_result[(df_result['result'] > 0)].shape[0] / len(df_result)
mean_n_api_call = df_result['n_of_api_call'].mean()
print('< summary >')
print(f"(n_episodes : {n_episodes}) accept_ratio: {accept_ratio: .3f}, excess_ratio: {excess_ratio:.3f}, mean_n_api_call: {mean_n_api_call:.2f}")

# df_result.to_csv(f"{env_path}/offline_test_result/{save_file_name_base}.csv", index=False, encoding='utf-8-sig')
# print('save test result.')





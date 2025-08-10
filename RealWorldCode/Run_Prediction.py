############################################################################################################
# (Path Setting)
import os
import sys
file_name = os.path.abspath(__file__)
file_path = os.path.dirname(file_name)
base_path = '/'.join(file_path.replace('\\','/').split('/')[:[i for i, d in enumerate(file_path.replace('\\','/').split('/')) if 'BaroNowProject' in d][0]+1])
file_path.split('/')[:[i for i, d in enumerate(file_path.split('/')) if 'BaroNowProject' in d][0]+1]
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

##########################################################################################################################################################
"""model_name <- Put your model's name here"""

from model.Algorithm4_Model import BaroNow_Algorithm4_Model as BaroNowModel
model_name = BaroNowModel.__name__
##########################################################################################################################################################


import sys
import select
sys.path.append(f'{base_path}')

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

# from tqdm.auto import tqdm
from datetime import datetime, timedelta

# from IPython.display import clear_output
# import time
import pytz
# import httpimport
import json
from six.moves import cPickle
# from functools import reduce
# import subprocess

from Run_Logging import logSave
kst = pytz.timezone('Asia/Seoul')


############################################################################################################
class BaroNowPrediction():
    def __init__(self, model='resnet', alpha=4, beta=1, gamma=2.5, max_api_call=None):
        """
        model : (str) resnet / transformer
        alpha : (float) prediction model sigma coefficient
        beta : (float) dynamic model sigma coefficient
        gamma : (float) prediction model discount factor
        """
        # self.model = BaroNowModel(load_info='pkl')
        self.model = BaroNowModel(model=model, load_info='json')

        self.alpha = alpha      # prediction model sigma coefficient
        self.beta = beta        # dynamic model sigma coefficient
        self.gamma = gamma      # prediction model discount factor
        self.max_api_call = max_api_call

    def predict(self, response_env, mode='online', alpha=None, beta=None, gamma=None, max_api_call=None, **kwargs):
        """
        response_env['contexts'] = [{...}, {...}, ...]
        response_env['status'] = 0/1/2  # 0 : 진행, 1: 정상종료, 2: New Episode
        """
        alpha = alpha if alpha is not None else self.alpha
        beta = beta if beta is not None else self.beta
        gamma = gamma if gamma is not None else self.gamma
        max_api_call = max_api_call if max_api_call is not None else self.max_api_call

        output = {}
        log_dict = {}          # log info 저장을 위한 공간
        special_event = {}

        contexts = response_env['contexts']
        contexts_df = pd.DataFrame(contexts)
        n_of_api_call = len(contexts_df['call_time_LastAPI'].dropna().drop_duplicates())

        cur_context = contexts[-1]
        if (max_api_call is not None) and (n_of_api_call >= max_api_call):
            response_model = self.model.predict(contexts, alpha=0, beta=beta, gamma=np.inf)       # model로 context 전달 및 model실행
        else:
            response_model = self.model.predict(contexts, alpha=alpha, beta=beta, gamma=gamma)       # model로 context 전달 및 model실행

        cur_time = datetime.strptime(cur_context['cur_time'], "%Y-%m-%dT%H:%M:%S%z")
                
        # output : environment에 전달할 내용 ---------------------------------------------------------------------------------------
        output['api_call_time'] = None
        output['leaving_time'] = None
        output['new_episodes'] = response_model['new_episodes']

        # max api_call 제한
        if (max_api_call is not None) and (n_of_api_call >= max_api_call):
            output['leaving_time'] = response_model['check_time']
            output['new_episodes'] = response_model['new_episodes']

        # no API Call
        elif pd.isna(cur_context['path_time_LastAPI']) is True:
            output['api_call_time'] = response_model['check_time']
            output['new_episodes'] = response_model['new_episodes']

        # API Call 이력이 있었던 경우
        else:
            call_time_LastAPI = datetime.strptime(cur_context['call_time_LastAPI'], "%Y-%m-%dT%H:%M:%S%z")

            # api call time시점에서 출발했을 때 target_time -13분 이내인 경우
            if call_time_LastAPI + timedelta(seconds=cur_context['path_time_LastAPI']) >= datetime.strptime(cur_context['target_time'], "%Y-%m-%dT%H:%M:%S%z") - timedelta(seconds=int(13*60)):
                output['leaving_time'] = cur_context['call_time_LastAPI']       # 바로 출발

                if response_model['logs']['dynamic_distance'] >= 4:
                    beta = response_model['logs']['dynamic_beta']
                    dynamic_mu, dynamic_std = response_model['logs']['pred_dynamic']
                    new_episode_time = call_time_LastAPI - timedelta(seconds=dynamic_mu*int(60)) - timedelta(seconds=beta*dynamic_std*int(60))
                    output['new_episodes'] = datetime.strftime(new_episode_time, "%Y-%m-%dT%H:%M:%S%z")
                else:
                    output['new_episodes'] = cur_context['call_time_LastAPI']
                
            else:
                check_time_datetime = datetime.strptime(response_model['check_time'], "%Y-%m-%dT%H:%M:%S%z")
                last_api_call_time = datetime.strptime(cur_context['call_time_LastAPI'], "%Y-%m-%dT%H:%M:%S%z")
                minimum_next_api_call_time = last_api_call_time + timedelta(seconds=int(180))

                # 3 Minute Rule
                if minimum_next_api_call_time > check_time_datetime:
                    
                    output['api_call_time'] = datetime.strftime(minimum_next_api_call_time, "%Y-%m-%dT%H:%M:%S%z")
                    beta = response_model['logs']['dynamic_beta']
                    dynamic_mu, dynamic_std = response_model['logs']['pred_dynamic']
                    new_episode_time = minimum_next_api_call_time - timedelta(seconds=dynamic_mu*int(60)) - timedelta(seconds=beta*dynamic_std*int(60))
                    output['new_episodes'] = datetime.strftime(new_episode_time, "%Y-%m-%dT%H:%M:%S%z")

                    # print(f'\t3 minute rule : {last_api_call_time} / {check_time_datetime}')
                    special_event['time'] = datetime.strftime(cur_time, "%Y-%m-%dT%H:%M:%S%z")
                    special_event['event'] = '3 minute rule.'
                else:
                    output['api_call_time'] = response_model['check_time']  # API Call 시간 Return
                    output['new_episodes'] = response_model['new_episodes']


                # now_date = datetime.now(kst)
                # # 3 Minute Rule
                # if (now_date - last_api_call_time).total_seconds() < 180:
                #     next_api_call_time = last_api_call_time + timedelta(seconds=int(180))
                #     output['api_call_time'] = datetime.strftime(next_api_call_time, "%Y-%m-%dT%H:%M:%S%z")
                #     beta = response_model['logs']['dynamic_beta']
                #     dynamic_mu, dynamic_std = response_model['logs']['pred_dynamic']
                #     new_episode_time = next_api_call_time - timedelta(seconds=dynamic_mu*int(60)) - timedelta(seconds=beta*dynamic_std*int(60))
                #     output['new_episodes'] = datetime.strftime(new_episode_time, "%Y-%m-%dT%H:%M:%S%z")
                # else:
                #     output['api_call_time'] = response_model['check_time']  # API Call 시간 Return
                #     output['new_episodes'] = response_model['new_episodes']
        

        # logging --------------------------------------------------------------------------------------------------------------
        if mode == 'online':
            response_env_copy = response_env.copy()
            response_env_copy['contexts'] = response_env['contexts'][-1:]
            
            log_dict['USER'] = response_env['id']
            log_dict['Response_Env_History'] = response_env_copy
            log_dict['Response_Model_History'] = response_model
            log_dict['Special_Event_History'] = special_event

            log = logSave("logging", logname=f"{model_name}")
            log.LogTextOut(log_dict)

            """Delete all log files in the logging directory."""
            # log.deleteAllLogs()
        # --------------------------------------------------------------------------------------------------------------------

        # # 종료조건
        # if response_env['status'] == 1:
        #     # Performance Evaluation ----------------------------------------------------------------------------------------
        #     last_context = pd.DataFrame(response_env['contexts']).iloc[-1,:]
        #     if not pd.isna(last_context['path_time_LastAPI']):
        #         arrival_time = datetime.strptime(last_context['cur_time'], "%Y-%m-%dT%H:%M:%S%z") + timedelta(seconds=last_context['path_time_LastAPI'])
        #         residual_arrival_time = (arrival_time - datetime.strptime(last_context['target_time'], "%Y-%m-%dT%H:%M:%S%z") ).total_seconds()/60
        #         # print(f"Is in allowance range? : {-13 < residual_arrival_time < 0}, {residual_arrival_time}")

        # Run_Prediction Output Environment로 전달
        return output


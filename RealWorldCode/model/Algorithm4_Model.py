############################################################################################################
# (Path Setting)
import os
import sys
file_name = os.path.abspath(__file__)
file_path = os.path.dirname(file_name)
base_path = '/'.join(file_path.replace('\\','/').split('/')[:[i for i, d in enumerate(file_path.replace('\\','/').split('/')) if 'BaroNowProject' in d][0]+1])
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
############################################################################################################

# print(f" *** Load Library *** ")

import json


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datetime import datetime, timedelta

from IPython.display import clear_output
import time
from tqdm.auto import tqdm
import pytz
# import httpimport

from six.moves import cPickle
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# customized module
from module.Module_Preprocessing import datetime_encoding, BaroNowPreprocessing
from module.Module_Utils import tensor_to_list, list_to_tensor
from module.Module_TorchModel import BaroNowResNet_V2, BaroNowTransformer_V2
from module.Module_TorchModel import BaroNowResNet_V3, BaroNowTransformer_V3

from geopy.distance import geodesic
import holidays
import pickle
import copy
##############################################################################################################


class BaroNow_Algorithm4_Model():
    def __init__(self, target_time_safe_margin=8*60, model='transformer', load_info='json'):
        """
        model = resnet / transformer
        """
                
        self.kst = pytz.timezone('Asia/Seoul')
        self.now_date = datetime.strftime(datetime.now(self.kst), "%Y-%m-%dT%H:%M:%S%z")
        self.target_time_safe_margin = target_time_safe_margin
        self.device = 'cpu'
        self.model = model
        self.dynamic_beta = 2

        # self.base_path = '/home/kimds929/BaroNowProject_V3'
        self.base_path = base_path
        self.weight_path = f'{self.base_path}/weight'
        self.preprocessing_instance = BaroNowPreprocessing()

        self.model_dict = {}
        self.model_dict['weights_file_name'] = None
        if self.model == 'resnet':
            self.model_dict['weights_file_name'] = 'Real_Alg4_pathtime_var_weights_ResNet_V3'
        elif self.model == 'transformer':
            self.model_dict['weights_file_name'] = 'Real_Alg4_pathtime_var_weights_Transformer_V3'
        self.model_dict['model'] = None
        self.model_dict['meta_data'] = None
        self.model_dict['weights'] = None
        self.model_dict['hyper_params'] = {}
        self.model_dict['feature_columns'] = {}

        self.load_info = load_info
        self.load_model_info()
        self.load_model()

        self.log = {}
        self.reset_outputs()

    # reset_outputs
    def reset_outputs(self):
        self.log = {
                'check_time': None,
                'new_episodes': None,
                'logs': {}
                }
        self.log['logs'] = {}
        self.log['logs']['timemachine_arrival_time'] = None
        self.log['logs']['pred_alpha'] = None
        self.log['logs']['pred_gamma'] = None
        self.log['logs']['pred_apply_alpha'] = None
        self.log['logs']['pred_tm'] = None
        self.log['logs']['pred_api'] = None
        self.log['logs']['pred_dynamic'] = None
        self.log['logs']['dynamic_beta'] = None
        self.log['logs']['dynamic_distance'] = None

    # load_model_info
    def load_model_info(self):
        if self.load_info == 'json':
            # (JSON) Load Model Info 
            with open(f"{self.weight_path}/{self.model_dict['weights_file_name']}.json", "r") as file:
                meta_data = json.load(file)

            # meta_data_var['header']
            self.model_dict['meta_data'] = meta_data
            self.model_dict['weights'] = list_to_tensor(meta_data['body']['weights'])
            self.model_dict['hyper_params'] = meta_data['body']['hyper_params']
            self.model_dict['feature_columns'] = meta_data['body']['feature_columns']
            self.model_dict['feature_columns']['dynamic'] = copy.deepcopy(self.model_dict['feature_columns']['base'])
            self.model_dict['feature_columns']['dynamic']['spatial_cols'][-1] = 'cur_point'

    # load_model
    def load_model(self):
        if self.model == 'resnet':
            # self.model_dict['model'] = model = BaroNowResNet_V2(**self.model_dict['hyper_params'])
            self.model_dict['model'] = model = BaroNowResNet_V3(**self.model_dict['hyper_params'])
        elif self.model == 'transformer':
            # self.model_dict['model'] = model = BaroNowTransformer_V2(**self.model_dict['hyper_params'])
            self.model_dict['model'] = model = BaroNowTransformer_V3(**self.model_dict['hyper_params'])
        self.model_dict['model'].load_state_dict(self.model_dict['weights'])
        self.model_dict['model'].to(self.device)
        self.model_dict['model'].eval()

    # model_predict
    def model_predict(self, cur_context_df, pred_type='dynamic'):
        X_list = []

        if pred_type == 'dynamic':
            X1 = self.preprocessing_instance.preprocessing(cur_context_df, self.model_dict['feature_columns']['dynamic'])
        else:
            X1 = self.preprocessing_instance.preprocessing(cur_context_df, self.model_dict['feature_columns']['base'])
        X_list.append(X1)

        if pred_type in ['api_call', 'leaving_time']:
            X2 = self.preprocessing_instance.preprocessing(cur_context_df, self.model_dict['feature_columns']['timemachine'])
            X_list.append(X2)
        
        if pred_type == 'leaving_time':
            X3 = self.preprocessing_instance.preprocessing(cur_context_df, self.model_dict['feature_columns']['api'])
            X_list.append(X3)

        return self.model_dict['model'](*X_list)

    # predict (★ 실제 모델에서 call하는 method)
    def predict(self, contexts_dict, alpha=3, beta=1, gamma=2):
        self.reset_outputs()
        contexts_df = self.preprocessing_instance.type_transform(pd.DataFrame(contexts_dict))
        n_of_api_call = len(contexts_df['call_time_LastAPI'].dropna().drop_duplicates())

        cur_context_df = contexts_df.iloc[[-1],:]
        timemachine_arrival_time = cur_context_df['target_time'].item() - timedelta(seconds=cur_context_df['path_time_TimeMachine'].item()) - timedelta(seconds=self.target_time_safe_margin)
        self.log['logs']['timemachine_arrival_time'] = datetime.strftime(timemachine_arrival_time, "%Y-%m-%dT%H:%M:%S%z")

        # api_call ---------------------------------------------------------------------------------
        if pd.isna(cur_context_df['call_time_LastAPI'].item()):
            mu, std = self.model_predict(cur_context_df, pred_type='api_call')
            mu, std = mu.item(), std.item()
            self.log['logs']['pred_tm'] = (mu, std)

        # leaving_time -----------------------------------------------------------------------------
        else:
            mu, std = self.model_predict(cur_context_df, pred_type='leaving_time')
            mu, std = mu.item(), std.item()
            self.log['logs']['pred_api'] = (mu, std)

        # checking time -----------------------------------------------------------------------------
        apply_alpha = alpha / gamma**(n_of_api_call)
        check_time = timemachine_arrival_time - timedelta(seconds=mu*60) - timedelta(seconds=apply_alpha*std*60)
        self.log['check_time'] = datetime.strftime(check_time, "%Y-%m-%dT%H:%M:%S%z")
        self.log['logs']['pred_alpha'] = alpha
        self.log['logs']['pred_gamma'] = gamma
        self.log['logs']['pred_apply_alpha'] = apply_alpha

        # new_episode (dynamic) --------------------------------------------------------------------
        dyanmic_distance = geodesic( cur_context_df['cur_point'].apply(lambda x: eval(x)[:2]).item(),
                cur_context_df['start_point'].apply(lambda x: eval(x)[:2]).item() ).kilometers
        
        dynamic_mu, dynamic_std = self.model_predict(cur_context_df, pred_type='dynamic')
        dynamic_mu, dynamic_std = dynamic_mu.item(), dynamic_std.item()
        self.log['logs']['pred_dynamic'] = (dynamic_mu, dynamic_std)
        new_episode_time = check_time - timedelta(seconds=dynamic_mu*60) - timedelta(seconds=beta*dynamic_std*60)
        self.log['logs']['dynamic_beta'] = beta

        self.log['new_episodes'] = datetime.strftime(new_episode_time, "%Y-%m-%dT%H:%M:%S%z") if dyanmic_distance >= 4 else datetime.strftime(check_time, "%Y-%m-%dT%H:%M:%S%z")
        self.log['logs']['dynamic_distance'] = dyanmic_distance

        output = self.log
        return output






# # ##################################################################################################
# # example ###
# def baronow_offline_simulator(contexts_data, episode_data, model_name='resnet'):
#     ba4_model = BaroNow_Algorithm4_Model(model=model_name)
#     # ba4_model = BaroNow_Algorithm4_Model(model='resnet')
#     ba4_model.predict(contexts_data)

#     episode_data_transform = ba4_model.preprocessing_instance.type_transform(episode_data)
#     episode_data_transform_copy = episode_data_transform.iloc[1:]
#     episode_data_transform_copy = episode_data_transform_copy.set_index('cur_time')
#     # episode_data_transform_copy.apply(lambda x: (x['target_time'] - x['call_time_LastAPI'] - timedelta(seconds=x['path_time_LastAPI'])).total_seconds()/60, axis=1)

#     episode_data_copy = episode_data.iloc[1:]
#     check_time = datetime.strptime(ba4_model.log['check_time'], "%Y-%m-%dT%H:%M:%S%z")

#     add_episode_data = episode_data_copy[(episode_data_transform['cur_time'] > check_time) & (check_time + timedelta(seconds=2*60) > episode_data_transform['cur_time'])]
#     if len(add_episode_data) > 0:
#         add_episode_data = add_episode_data.iloc[[0]]
#         add_episode_data_transform = ba4_model.preprocessing_instance.type_transform(add_episode_data)
#         rev_time = add_episode_data_transform.apply(lambda x : (x['target_time'] - x['cur_time'] - timedelta(seconds=x['path_time_LastAPI'])).total_seconds()/60, axis=1).item()

#         return add_episode_data, rev_time
#     else:
#         return pd.DataFrame(), False


# # # --------------------------------------------------------------------------------------------
# data = pd.read_csv(f"{dataset_path}/20241001.csv", encoding='utf-8-sig')

# n_iter = 1
# history = []

# for e in tqdm(range(n_iter)):
#     print(f'------ n_iter : {e+1} ---------------------------------------------------------------------')
#     episode = {}
#     sampled_id = data['id'].drop_duplicates().sample().item()
#     episode['id'] = sampled_id
    
#     episode_data = data[data['id'] == sampled_id]
#     init_data = episode_data.iloc[[0],:]

#     for api_col in ['call_time_LastAPI', 'call_point_LastAPI', 'path_LastAPI']:
#         init_data[api_col] = np.nan

#     # episode_data.to_dict('records')

#     # --------------------------------------------------------------------------------------------
#     print('-- (ResNet) --')
#     contexts = init_data.to_dict('records')
#     # add_data = episode_data.iloc[[1]]
#     # add_data = episode_data.iloc[[2]]

#     add_episode_data = pd.DataFrame()
#     contexts_data = init_data.copy()

#     n_of_api_call = 0
#     while(True):
#         contexts_data = pd.concat([contexts_data, add_episode_data], axis=0)

#         # contexts = contexts_data.to_dict('records')

#         add_episode_data, rev_time = baronow_offline_simulator(contexts_data, episode_data, 'resnet')
#         print(rev_time, end=' : ')
#         if len(add_episode_data) == 0 and rev_time is False:
#             print('not have data')
#             episode['n_of_api_call_resnet'] = None
#             break
#         elif n_of_api_call > 10:
#             print('infitiy loop')
#             episode['n_of_api_call_resnet'] = np.inf
#             break
#         elif rev_time > 13:
#             print('additional api call')
#             n_of_api_call += 1
            
#         elif rev_time < 0:
#             print('excess!')
#             episode['n_of_api_call_resnet'] = n_of_api_call
#             episode['done_resnet'] = 'excess'
#             break
#         else:
#             print('arrival!')
#             print(f"  *** n_of_api_call: {n_of_api_call}")
#             episode['n_of_api_call_resnet'] = n_of_api_call
#             episode['done_resnet'] = 'arrival'
#             break
#     # --------------------------------------------------------------------------------------------

#     # --------------------------------------------------------------------------------------------
#     print('-- (Transformer) --')
#     contexts = init_data.to_dict('records')
#     # add_data = episode_data.iloc[[1]]
#     # add_data = episode_data.iloc[[2]]

#     add_episode_data = pd.DataFrame()
#     contexts_data = init_data.copy()

#     n_of_api_call = 0
#     while(True):
        
#         contexts_data = pd.concat([contexts_data, add_episode_data], axis=0)

#         # contexts = contexts_data.to_dict('records')

#         add_episode_data, rev_time = baronow_offline_simulator(contexts_data, episode_data, 'transformer')
#         print(rev_time, end=' : ')
#         if len(add_episode_data) == 0 and rev_time is False:
#             print('not have data')
#             episode['n_of_api_call_transformer'] = None
#             break
#         elif n_of_api_call > 10:
#             print('infitiy loop')
#             episode['n_of_api_call_transformer'] = np.inf
#             break
#         elif rev_time > 13:
#             print('additional api call')
#             n_of_api_call += 1
#         elif rev_time < 0:
#             print('excess!')
#             episode['n_of_api_call_transformer'] = n_of_api_call
#             episode['done_transformer'] = 'excess'
#             break
#         else:
#             print('arrival!')
#             print(f"  *** n_of_api_call: {n_of_api_call}")
#             episode['n_of_api_call_transformer'] = n_of_api_call
#             episode['done_transformer'] = 'arrival'
#             break
#     # --------------------------------------------------------------------------------------------
#     history.append(episode)

# pd.DataFrame(history)

# pd.DataFrame(history)[['n_of_api_call_resnet','n_of_api_call_transformer']].mean()



# X1 = ba4_model.preprocessing_instance.preprocessing(contexts_data.iloc[[-1]], ba4_model.model_dict['feature_columns']['base'])
# X2 = ba4_model.preprocessing_instance.preprocessing(contexts_data.iloc[[-1]], ba4_model.model_dict['feature_columns']['timemachine'])
# X3 = ba4_model.preprocessing_instance.preprocessing(contexts_data.iloc[[-1]], ba4_model.model_dict['feature_columns']['api'])

# X1.shape
# X2.shape
# X3.shape
# ba4_model.model_dict['model'](X1)
# ba4_model.model_dict['model'](X1, X2)
# ba4_model.model_dict['model'](X1, X2, X3)



# ###################################################################################################################
# data = pd.read_csv(f"{dataset_path}/20241001.csv", encoding='utf-8-sig')

# sampled_id = data['id'].drop_duplicates().sample().item()
# episode_data = data[data['id'] == sampled_id]
# init_data = episode_data.iloc[[0],:]

# for api_col in ['call_time_LastAPI', 'call_point_LastAPI', 'path_LastAPI']:
#     init_data[api_col] = np.nan

# episode_data.to_dict('records')

# contexts = init_data.to_dict('records')
# # add_data = episode_data.iloc[[1]]
# # add_data = episode_data.iloc[[2]]

# add_episode_data = pd.DataFrame()
# contexts_data = init_data.copy()

# contexts_data
# contexts_data = pd.concat([contexts_data, add_episode_data], axis=0)

# contexts = contexts_data.to_dict('records')

# contexts_data['cur_time']
# contexts_data['cur_point']
# contexts_data['start_point']
# # --------------------------------------------------------------------------------------------

# ba4_model = BaroNow_Algorithm4_Model(model='resnet')
# # ba4_model = BaroNow_Algorithm4_Model(model='transformer')
# ba4_model.predict(contexts_data)
# # print(ba4_model.log)

# episode_data_transform = ba4_model.preprocessing_instance.type_transform(episode_data)
# episode_data_transform_copy = episode_data_transform.iloc[1:]
# episode_data_transform_copy = episode_data_transform_copy.set_index('cur_time')
# episode_data_transform_copy.apply(lambda x: (x['target_time'] - x['call_time_LastAPI'] - timedelta(seconds=x['path_time_LastAPI'])).total_seconds()/60, axis=1)

# episode_data_copy = episode_data.iloc[1:]
# check_time = datetime.strptime(ba4_model.log['check_time'], "%Y-%m-%dT%H:%M:%S%z")


# add_episode_data = episode_data_copy[(episode_data_transform['cur_time'] > check_time) & (check_time + timedelta(seconds=2*60) > episode_data_transform['cur_time'])]

# if len(add_episode_data) > 0:
#     add_episode_data = add_episode_data.iloc[[0]]
#     add_episode_data_transform = ba4_model.preprocessing_instance.type_transform(add_episode_data)
#     rev_time = add_episode_data_transform.apply(lambda x : (x['target_time'] - x['cur_time'] - timedelta(seconds=x['path_time_LastAPI'])).total_seconds()/60, axis=1).item()

#     print(rev_time, end=' : ')
#     if rev_time > 13:
#         print('additional api call')
#     elif rev_time < 0:
#         print('excess!')
#         # break
#     else:
#         print('arrival!')
#         # break
        
# else:
#     print('not have data')
#     # break
# ###################################################################################################################
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

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from tqdm.auto import tqdm
from datetime import datetime, timedelta    

from IPython.display import clear_output
import time
from tqdm.auto import tqdm
import pytz
import httpimport

from six.moves import cPickle
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# logging module
from Run_Logging import readLog, lastTwo, lastTwoContexts, logSave

# customized module
from module.Module_Preprocessing import datetime_encoding, real_make_feature_set_embedding
from module.Module_Utils import tensor_to_list, list_to_tensor
from module.Module_TorchModel import CombinedEmbedding, EnsembleCombinedModel
from model.Algorithm3_Model import BaroNow_Algorithm3_Model as BaroNowModel

kst = pytz.timezone('Asia/Seoul')
baroNow_model = BaroNowModel()

class BatchUpdateModel:
    def __init__(self, load_info='json'):
        self.device = 'cpu'
        self.model_var = None
        self.model_pathtime = None
        self.var_train_loader = None
        self.pathtime_train_loader = None
        
        self.model_dict = {}
        self.transportation = None
        self.kinds_of_transportation = ['car_kakao', 'car_tmap']
        
        self.model_dict['car_kakao'] = {}
        self.model_dict['car_kakao']['var_weights_name'] = "Real_Alg3_offline_model_var_weights_car_kakao"
        self.model_dict['car_kakao']['pathtime_weights_name'] = "Real_Alg3_offline_model_pathtime_weights_car_kakao"
        self.model_dict['car_kakao']['dynamic_weights_name'] = "Real_Dynamic_LinearModel_V1"
        
        self.model_dict['car_tmap'] = {}
        self.model_dict['car_tmap']['var_weights_name'] = "Real_Alg3_offline_model_var_weights_car_tmap"
        self.model_dict['car_tmap']['pathtime_weights_name'] = "Real_Alg3_offline_model_pathtime_weights_car_tmap"
        self.model_dict['car_tmap']['dynamic_weights_name'] = "Real_Dynamic_LinearModel_V1"

        for transportation in self.kinds_of_transportation:
            self.model_dict[transportation]['var_model'] = None
            self.model_dict[transportation]['var_weights'] = None
            self.model_dict[transportation]['var_hyper_params'] = {}
            self.model_dict[transportation]['var_features'] = []
            self.model_dict[transportation]['var_train_loader'] = None
            
            self.model_dict[transportation]['pathtime_model'] = None
            self.model_dict[transportation]['pathtime_weights'] = None
            self.model_dict[transportation]['pathtime_hyper_params'] = {}
            self.model_dict[transportation]['pathtime_features'] = []
            self.model_dict[transportation]['pathtime_train_loader'] = None
            
            self.model_dict[transportation]['dynamic_model'] = None
            self.model_dict[transportation]['dynamic_weights'] = None
            self.model_dict[transportation]['dynamic_hyper_params'] = {}
            self.model_dict[transportation]['dynamic_features'] = []
            self.model_dict[transportation]['dynamic_train_loader'] = None
            
        self.load_info = load_info
        self.load_model_info()
        self.load_model()
        
        self.outputs = {}
        self.reset_outputs()
        
    def reset_outputs(self):
        self.outputs = {
                'api_call_time': None,
                'leaving_time': None,
                'new_episodes': 0,
                'logs': {}
                }
        self.outputs['logs'] = {}
        self.outputs['logs']['pred_std'] = None
        self.outputs['logs']['pred_delta'] = None
        self.outputs['logs']['dyanmic_pred_path_time'] = None
        self.outputs['logs']['dynamic_info'] = None
    
    def load_model_info(self):
        if self.load_info == 'json':
            for transportation in self.kinds_of_transportation:
                '''Load Variance Model'''
                if self.model_dict[transportation]['var_weights_name'] is not None:
                    with open(f"{weight_path}/{self.model_dict[transportation]['var_weights_name']}.json", "r") as file:
                        meta_data_var = json.load(file)
                    
                    self.model_dict[transportation]['var_weights'] = list_to_tensor(meta_data_var['body']['weights'])
                    self.model_dict[transportation]['var_hyper_params'] = meta_data_var['body']['hyper_params']
                    self.model_dict[transportation]['var_features'] = meta_data_var['body']['feature_columns'] # feature_columns_var
                    
                '''Load Pathtime Model'''
                if self.model_dict[transportation]['pathtime_weights_name'] is not None:
                    with open(f"{weight_path}/{self.model_dict[transportation]['pathtime_weights_name']}.json", "r") as file:
                        meta_data_pathtime = json.load(file)
                        
                    self.model_dict[transportation]['pathtime_weights'] = list_to_tensor(meta_data_pathtime['body']['weights'])
                    self.model_dict[transportation]['pathtime_hyper_params'] = meta_data_pathtime['body']['hyper_params']
                    self.model_dict[transportation]['pathtime_features'] = meta_data_pathtime['body']['feature_columns'] # feature_columns_pathtime

                '''Load Dynamic Model'''
                # if self.model_dict[transportation]['dynamic_weights_name'] is not None:
                #     with open(f"{self.weight_path}/{self.model_dict[transportation]['dynamic_weights_name']}.json", "r") as file:
                #         meta_data_dynamic = json.load(file)
                    
                #     self.model_dict[transportation]['dynamic_weights'] = eval(meta_data_dynamic['body']['weights'])
                    
    def load_model(self):   
        for transportation in self.kinds_of_transportation:
            '''Variance Model'''    
            self.model_dict[transportation]['var_model'] = EnsembleCombinedModel(**self.model_dict[transportation]['var_hyper_params'])
            self.model_dict[transportation]['var_model'].load_state_dict(self.model_dict[transportation]['var_weights']) 
            self.model_dict[transportation]['var_model'].to(self.device)

            '''Pathtime Model'''
            self.model_dict[transportation]['pathtime_model'] = EnsembleCombinedModel(**self.model_dict[transportation]['pathtime_hyper_params'])
            self.model_dict[transportation]['pathtime_model'].load_state_dict(self.model_dict[transportation]['pathtime_weights'])
            self.model_dict[transportation]['pathtime_model'].to(self.device)

            '''Dynamic Model'''
            # self.model_dict[transportation]['dynamic_model'] = BaroNow_DynamicModel()
            # self.model_dict[transportation]['dynamic_model'].load_state_dict(self.model_dict[transportation]['dynamic_weights'])  
        
    def create_dataloader(self, df_contexts, transportation, batch_size=64):
        '''Data Preprocessing'''
        df_contexts = baroNow_model.data_preprocessing(df_contexts.to_dict('records'))
        
        '''Variance Dataset'''
        df_var = df_contexts.iloc[[-1]].copy()
        columns_var = reduce(lambda x,y: x+y, list(self.model_dict[transportation]['var_features'].values()))
        context_var_transform = real_make_feature_set_embedding(df_var[columns_var], **self.model_dict[transportation]['var_features'])
        context_var_torch = torch.tensor(context_var_transform.to_numpy(), dtype=torch.float32)

        var_target = (df_var['path_time_TimeMachine'] - df_var['path_time_LastAPI']) / 60
        var_target_tensor = torch.tensor(var_target.to_numpy().reshape(-1,1), dtype=torch.float32)
        var_train_dataset = TensorDataset(context_var_torch, var_target_tensor)
        # self.var_train_loader = DataLoader(var_train_dataset, batch_size=batch_size, shuffle=True)
        self.model_dict[transportation]['var_train_loader'] = DataLoader(var_train_dataset, batch_size=batch_size, shuffle=True)
        
        '''Pathtime Dataset'''
        df_pathtime = df_contexts.iloc[[-2]].copy()
        columns_pathtime = reduce(lambda x,y: x+y, list(self.model_dict[transportation]['pathtime_features'].values()))
        context_pathtime_transform = real_make_feature_set_embedding(df_pathtime[columns_pathtime], **self.model_dict[transportation]['pathtime_features'])
        context_pathtime_torch = torch.tensor(context_pathtime_transform.to_numpy(), dtype=torch.float32)

        # print(df_var['path_time_LastAPI'].shape, df_pathtime['path_time_LastAPI'].shape)
        # pathtime_target = (df_pathtime['path_time_ans'] - df_pathtime['path_time_LastAPI']) / 60
        pathtime_target = (df_var['path_time_LastAPI'].reset_index(drop=True) - df_pathtime['path_time_LastAPI'].reset_index(drop=True)) / 60 # 2024.11.08 수정
        pathtime_target_tensor = torch.tensor(pathtime_target.to_numpy().reshape(-1, 1), dtype=torch.float32)

        # print(context_pathtime_torch.shape, pathtime_target_tensor.shape)
        pathtime_train_dataset = TensorDataset(context_pathtime_torch, pathtime_target_tensor)
        # self.pathtime_train_loader = DataLoader(pathtime_train_dataset, batch_size=batch_size, shuffle=True)
        self.model_dict[transportation]['pathtime_train_loader'] = DataLoader(pathtime_train_dataset, batch_size=batch_size, shuffle=True)
            
        '''Dynamic Dataset'''
        # columns_dynamic = reduce(lambda x,y: x+y, list(self.model_dict[transportation]['dynamic_features']))
            
    def run_batch_update(self, model_name, date, transportation, num_epochs=1):
        '''Varinace Offline Learning'''
        losses_var = []
        optimizer_var = optim.Adam(self.model_dict[transportation]['var_model'].parameters(), lr=1e-4)
        
        for epoch in range(num_epochs):
            self.model_dict[transportation]['var_model'].train()
            mu_mean = []
            std_mean = []
            loss_mean = []
            with tqdm(total=len(self.model_dict[transportation]['var_train_loader']), desc=f"Epoch {epoch+1}/{num_epochs}", ) as pbar:
                for bi, (batch_x, batch_y) in enumerate(self.model_dict[transportation]['var_train_loader']):
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    optimizer_var.zero_grad()
                    samples = self.model_dict[transportation]['var_model'](batch_x)
                    std = samples.std()
                    mu = torch.zeros_like(std)
                    loss = torch.nn.functional.gaussian_nll_loss(mu, batch_y, std**2)                    
                    
                    if torch.isnan(loss):
                        break
                    
                    loss.backward()
                    optimizer_var.step()

                    with torch.no_grad():
                        loss_mean.append(loss.item())
                        mu_mean.append(mu.item())
                        std_mean.append(std.item())
                    
                    if bi % 1 == 0:
                        pbar.set_postfix(Loss=f"{np.mean(loss_mean).item():.2f}", Mu=f"{np.mean(mu_mean).item():.2f}", Std=f"{np.mean(std_mean).item():.2f}")
                    pbar.update(1)

            with torch.no_grad():
                losses_var.append( np.mean(loss_mean).item() )
                # print(f'\rEpoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}, Mu:{torch.mean(mu).item:.2f}, std: {torch.mean(std).item():.2f}', end='')

                # -----------------------------------------------------------------
                # update break criteria
                if np.mean(loss_mean).item() < 5:
                    break
                # -----------------------------------------------------------------
                    
        '''Pathtime Offline Learning'''
        losses_pathtime = []
        optimizer_pathtime = optim.Adam(self.model_dict[transportation]['pathtime_model'].parameters(), lr=1e-5)
        
        for epoch in range(num_epochs):
            self.model_dict[transportation]['pathtime_model'].train()
            pred_delta_mean = []
            loss_mean = []
            with tqdm(total=len(self.model_dict[transportation]['pathtime_train_loader']), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for bi, (batch_x, batch_y) in enumerate(self.model_dict[transportation]['pathtime_train_loader']):
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    optimizer_pathtime.zero_grad()
                    pred_delta = self.model_dict[transportation]['pathtime_model'](batch_x)
                    loss = torch.nn.functional.mse_loss(pred_delta, batch_y)
                    
                    if torch.isnan(loss):
                        break
                        
                    loss.backward()
                    optimizer_pathtime.step()
                    
                    with torch.no_grad():
                        loss_mean.append(loss.item())
                        pred_delta_mean.append(torch.mean(pred_delta).item())
                    
                    if bi % 30 == 0:
                        pbar.set_postfix(Loss=f"{np.mean(loss_mean).item():.2f}", Mean=f"{np.mean(pred_delta_mean).item():.2f}")
                    pbar.update(1)

            with torch.no_grad():
                losses_pathtime.append(np.mean(loss_mean).item())
                # print(f'\rEpoch {epoch+1}/{num_epochs}, Loss: {np.mean(loss_mean).item():.2f}, mean: {np.mean(pred_delta_mean).item():.2f}', end='')

                # -----------------------------------------------------------------
                # update break criteria
                if np.mean(loss_mean).item() < 25:
                    break
                # -----------------------------------------------------------------
                
        '''Dynamic Offline Learning'''
        
        '''Save BatchedUpdate History to json'''
        history_folder = f"{weight_path}/batch_log"
        history_save_name = "BatchedUpdateHistory"
        os.makedirs(history_folder, exist_ok=True)
        history_file_path = f"{history_folder}/{history_save_name}.json"
        
        batched_update_history = {
            'model_name': model_name,
            'date': date,
            'transportation': transportation,
            'num_epochs': num_epochs,
            'losses_var': losses_var,
            'losses_pathtime': losses_pathtime
        }
        
        if os.path.exists(history_file_path):
            with open(history_file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            date_exists = any(entry['date'] == date for entry in existing_data)
            
            if date_exists:
                for entry in existing_data:
                    if entry['date'] == date:
                        entry['losses_var'] = losses_var
                        entry['losses_pathtime'] = losses_pathtime
                        break
            else:
                existing_data.append(batched_update_history)

            with open(history_file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
        else:
            with open(history_file_path, 'w', encoding='utf-8') as f:
                json.dump([batched_update_history], f, indent=4, ensure_ascii=False)
    
    def save_weights(self, transportation):
        now_date = datetime.strftime(datetime.now(kst), "%Y-%m-%dT%H:%M:%S%z")
        today = datetime.strftime(datetime.now(kst), "%Y-%m-%d")
        var_base = f"Real_Alg3_offline_model_var_weights_{transportation}"
        pathtime_base = f"Real_Alg3_offline_model_pathtime_weights_{transportation}"
        
        '''1. Save Variance Model Meta Data'''
        var_meta_data = baroNow_model.model_dict[transportation]['var_meta_data']
        var_meta_data['header']['last_update'] = now_date
        var_meta_data['header']['path'] = f"{weight_path}/{var_base}.pkl"
        var_meta_data['header']['source'].append(today)
        
        '''Convert model state dict to a list'''
        state_dict = self.model_dict[transportation]['var_model'].state_dict()
        state_dict_json = tensor_to_list(state_dict)
        var_meta_data['body']['weights'] = state_dict_json

        '''To pickle and json'''
        var_save_name = f"{today}_{var_base}"     
        with open (f"{weight_path}/weight_log/{var_save_name}.pkl", 'wb') as f: # 날짜O
            cPickle.dump(var_meta_data, f)
        with open(f"{weight_path}/{var_base}.pkl", 'wb') as f: # 날짜X (덮어씀)
            cPickle.dump(var_meta_data, f)
        with open(f"{weight_path}/weight_log/{var_save_name}.json", 'w', encoding='utf-8') as f:
            json.dump(var_meta_data, f, indent=4, ensure_ascii=False)
        with open(f"{weight_path}/{var_base}.json", 'w', encoding='utf-8') as f:
            json.dump(var_meta_data, f, indent=4, ensure_ascii=False)
        
        '''2. Save Pathtime Model Meta Data'''
        pathtime_meta_data = baroNow_model.model_dict[transportation]['pathtime_meta_data']
        pathtime_meta_data['header']['last_update'] = now_date
        pathtime_meta_data['header']['path'] = f"{weight_path}/{pathtime_base}.pkl"
        pathtime_meta_data['header']['source'].append(today)
        
        '''Convert model state dict to a list'''
        state_dict = self.model_dict[transportation]['pathtime_model'].state_dict()
        state_dict_json = tensor_to_list(state_dict)
        pathtime_meta_data['body']['weights'] = state_dict_json    
        
        '''To pickle and json''' # weight_log 폴더에 날짜 붙은 것이 쌓임(기록들)
        pathtime_save_name = f"{today}_{pathtime_base}"
        with open(f"{weight_path}/weight_log/{pathtime_save_name}.pkl", 'wb') as f:
            cPickle.dump(pathtime_meta_data, f)
        with open(f"{weight_path}/{pathtime_base}.pkl", 'wb') as f:
            cPickle.dump(pathtime_meta_data, f)
        with open(f"{weight_path}/weight_log/{pathtime_save_name}.json", 'w', encoding='utf-8') as f:
            json.dump(pathtime_meta_data, f, indent=4, ensure_ascii=False)
        with open(f"{weight_path}/{pathtime_base}.json", 'w', encoding='utf-8') as f:
            json.dump(pathtime_meta_data, f, indent=4, ensure_ascii=False)
            
        print(f"Weights and metadata for '{transportation}' saved successfully.")
        
        '''3. Save Dynamic Model Meta Data'''

##############################################################################################################
'''
transportation = 'car_tmap'
now_date = datetime.strftime(datetime.now(kst), "%Y-%m-%dT%H:%M:%S%z")
today = datetime.strftime(datetime.now(kst), "%Y-%m-%d")


# model = BaroNowModel()
# model.model_dict[transportation]['var_meta_data']
# model.model_dict[transportation]['var_meta_data']['header']
# model.model_dict[transportation]['var_meta_data']['body']

# meta_data_weight = {
#         "header": {
#             "title": None,
#             "description":None,
#             "last_update": now_date,    # ★
#             "path": None,   # ★
#             "source": []       # ★
#         },
#         "body": {
#             "feature_columns" : {},
#             "hyper_params": [],
#             "model_structures": [],
#             "weights": []   # ★
#         }
#     }

# (var) meta data --------------------------------------------------------------------------------------------
save_file_name_base = f"Real_Alg3_offline_model_var_weights_{transportation}"
save_file_name = f"{now_date[:10]}_" + save_file_name_base

var_meta_data = model.model_dict[transportation]['var_meta_data']
var_meta_data['header']['last_update'] = now_date     # ★
var_meta_data['header']['path'] = f"{weight_path}/{save_file_name}.pkl"  # ★
var_meta_data['header']['source'] = model.model_dict[transportation]['var_meta_data']['header']['source'] + [today]     # ★

state_dict = model_var.state_dict()            # 학습한 모델.state_dict()
state_dict_json = tensor_to_list(state_dict)        # tensor_to_list 함수 활용해서 tensor를 list로
var_meta_data['body']['weights'] = state_dict_json           # ★

# save to pickle
cPickle.dump(var_meta_data, open(f"{weight_path}/weight_log/{save_file_name}.pkl", 'wb'))     # 날짜 들어간거
cPickle.dump(var_meta_data, open(f"{weight_path}/{save_file_name_base}.pkl", 'wb'))    # 날짜 안들어간거

# save info to json
with open(f"{weight_path}/weight_log/{save_file_name}.json", 'w', encoding='utf-8') as file:        # 날짜 들어간거
    json.dump(var_meta_data, file, indent=4, ensure_ascii=False) 
with open(f"{weight_path}/{save_file_name_base}.json", 'w', encoding='utf-8') as file:      # 날짜 안들어간거
    json.dump(var_meta_data, file, indent=4, ensure_ascii=False) 

# (pathtime) meta data --------------------------------------------------------------------------------------------
save_file_name_base = f"Real_Alg3_offline_model_pathtime_weights_{transportation}"
save_file_name = f"{now_date[:10]}_" + save_file_name_base

pathtime_meta_data = model.model_dict[transportation]['pathtime_meta_data']
pathtime_meta_data['header']['last_update'] = now_date     # ★
pathtime_meta_data['header']['path'] = f"{weight_path}/{save_file_name}.pkl"  # ★
pathtime_meta_data['header']['source'] = model.model_dict[transportation]['pathtime_meta_data']['header']['source'] + [today]     # ★

state_dict = model_pathtime.state_dict()            # 학습한 모델.state_dict()
state_dict_json = tensor_to_list(state_dict)        # tensor_to_list 함수 활용해서 tensor를 list로
pathtime_meta_data['body']['weights'] = state_dict_json           # ★

# save to pickle 
cPickle.dump(pathtime_meta_data, open(f"{weight_path}/weight_log/{save_file_name}.pkl", 'wb'))     # 날짜 들어간거
cPickle.dump(pathtime_meta_data, open(f"{weight_path}/{save_file_name_base}.pkl", 'wb'))    # 날짜 안들어간거

# save info to json
with open(f"{weight_path}/weight_log/{save_file_name}.json", 'w', encoding='utf-8') as file:        # 날짜 들어간거
    json.dump(pathtime_meta_data, file, indent=4, ensure_ascii=False) 
with open(f"{weight_path}/{save_file_name_base}.json", 'w', encoding='utf-8') as file:      # 날짜 안들어간거
    json.dump(pathtime_meta_data, file, indent=4, ensure_ascii=False) 
'''
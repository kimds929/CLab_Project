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
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# logging module
from Run_Logging import readLog, lastTwo, lastTwoContexts, logSave

# customized module
from module.Module_Preprocessing import datetime_encoding, BaroNowPreprocessing
from module.Module_Utils import tensor_to_list, list_to_tensor
from module.Module_TorchModel import CombinedEmbedding 
from module.Module_TorchModel import BaroNowResNet_V2, BaroNowTransformer_V2, BaroNowResNet_V3, BaroNowTransformer_V3
from model.Algorithm4_Model import BaroNow_Algorithm4_Model as BaroNowModel

kst = pytz.timezone('Asia/Seoul')
baroNow_model4 = BaroNowModel()

class BatchUpdateModel4():
    def __init__(self, model='resnet', load_info='json'):
        self.kst = pytz.timezone('Asia/Seoul')
        self.now_date = datetime.strftime(datetime.now(self.kst), "%Y-%m-%dT%H:%M:%S%z")
        self.today = datetime.strftime(datetime.now(self.kst), "%Y-%m-%d")
        self.device = 'cpu'
        
        self.model = model # resnet or transformer (string)
        self.preprocessing = BaroNowPreprocessing()
        
        self.model_dict = {}
        self.model_dict['weights_file_name'] = None
        if self.model == 'resnet':
            self.model_dict['weights_file_name'] = 'Real_Alg4_pathtime_var_weights_ResNet_V3'
        elif self.model == 'transformer':
            self.model_dict['weights_file_name'] = 'Real_Alg4_pathtime_var_weights_Transformer_V3'
            
        self.load_info = load_info
        self.load_model_info()
        self.load_model()
        
        self.outputs = {}
        self.reset_outputs()

    def reset_outputs(self):
        self.outputs = {
            'check_time': None,
            'new_episodes': None,
            'logs': {}
        }
        self.log = {}
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
    
    def load_model_info(self):
        if self.load_info == 'json':
            with open(f"{weight_path}/{self.model_dict['weights_file_name']}.json", "r") as file:
                meta_data = json.load(file)
            
            self.model_dict['meta_data'] = meta_data
            self.model_dict['weights'] = list_to_tensor(meta_data['body']['weights'])
            self.model_dict['hyper_params'] = meta_data['body']['hyper_params']
            self.model_dict['feature_columns']  = meta_data['body']['feature_columns']
            self.model_dict['feature_columns']['dynamic'] = copy.deepcopy(self.model_dict['feature_columns']['base'])
            self.model_dict['feature_columns']['dynamic']['spatial_cols'][-1] = 'cur_point'

    def load_model(self):
        if self.model == 'resnet':
            self.model_dict['model'] = model = BaroNowResNet_V3(**self.model_dict['hyper_params'])
        elif self.model == 'transformer':
            self.model_dict['model'] = model = BaroNowTransformer_V3(**self.model_dict['hyper_params'])
        self.model_dict['model'].load_state_dict(self.model_dict['weights'])
        self.model_dict['model'].to(self.device)
        self.model_dict['model'].eval()
    
    def run_batch_update(self, algorithm, df, index_loader, feature_dicts={}, pred_label='optimal_delta'):
        '''Offline Learning w/ Random Sampling''' 
        batch_gaussian_loss = []
        batch_mse_loss = []
        batch_delta_mean = []
        batch_delta_std = []
        
        model = self.model_dict['model']
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        
        with tqdm(total=len(index_loader)) as pbar:
            model.train()
            
            for bi, (batch_idx) in enumerate(index_loader):
                batch_idx = batch_idx[0].numpy() 
                
                batch_data = df.loc[batch_idx]  
                batch_x_dict = {}
                for arg_name, feature_dict in feature_dicts.items():
                    batch_x_dict[arg_name] = self.preprocessing.preprocessing(data=batch_data, feature_dict=feature_dict).to(self.device)
                batch_y = (torch.tensor(batch_data[pred_label].to_numpy().reshape(-1,1)).type(torch.float32)/60).to(self.device)
                
                pred_mu = model(**batch_x_dict, pred_type='mu')
                mse_loss = nn.functional.mse_loss(pred_mu, batch_y)

                optimizer.zero_grad()
                mse_loss.backward()
                optimizer.step()
                
                pred_std = model(**batch_x_dict, pred_type='std')
                pred_mu_ = pred_mu.detach()
                gaussian_loss = nn.functional.gaussian_nll_loss(pred_mu_, batch_y, pred_std**2)
                
                optimizer.zero_grad()
                gaussian_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    batch_gaussian_loss.append(gaussian_loss.item())
                    batch_mse_loss.append(mse_loss.item())
                    batch_delta_mean.append(torch.mean(pred_mu).item())
                    batch_delta_std.append(torch.mean(pred_std).item())
                
                if bi % 1 == 0:
                    pbar.set_postfix(Gaussian_Loss=f"{np.mean(batch_gaussian_loss).item():.2f}", RMSE_Loss=f"{np.sqrt(np.mean(batch_mse_loss)).item():.2f}",
                                     Mean=f"{np.mean(batch_delta_mean).item():.2f}", Std=f"{np.mean(batch_delta_std).item():.2f}")
                pbar.update(1)

        '''Save BatchedUpdate History to json'''
        history_folder = f"{weight_path}/batch_log"
        history_save_name = "BatchedUpdateHistory"
        os.makedirs(history_folder, exist_ok=True)
        history_file_path = f"{history_folder}/{history_save_name}"
        
        batched_update_history = {
            'algorithm': algorithm,
            'model': self.model,
            'date': self.today,
            'feature': feature_dicts,
            'batch_gaussian_loss': batch_gaussian_loss,
            'batch_mse_loss': batch_mse_loss
        }
        
        if os.path.exists(history_file_path):
            with open(history_file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            already_exists = None
            for entry in existing_data:
                if entry['date'] == self.today and json.dumps(entry['feature'], sort_keys=True) == json.dumps(feature_dicts, sort_keys=True):
                    already_exists = entry
                    break
                        
            if already_exists:
                already_exists['algorithm'] = algorithm
                already_exists['model'] = self.model
                already_exists['batch_gaussian_loss'] = batch_gaussian_loss
                already_exists['batch_mse_loss'] = batch_mse_loss
            else:
                existing_data.append(batched_update_history)
            
            with open(history_file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
        else:
            with open(history_file_path, 'w', encoding='utf-8') as f:
                json.dump([batched_update_history], f, indent=4, ensure_ascii=False)
                       
    def save_weights(self):
        base = f"Real_Alg4_offline_pathtime_var_weights_{self.model}"
        save_name = f"{self.today}_{base}"
        
        meta_data = self.model_dict['meta_data']
        meta_data['header']['last_update'] = self.now_date
        meta_data['header']['path'] = f"{weight_path}/{base}.pkl"
        meta_data['header']['source'].append(self.today)
        
        '''Convert model state dict -> list'''
        state_dict = self.model_dict['model'].state_dict()
        state_dict_json = tensor_to_list(state_dict)
        meta_data['body']['weights'] = state_dict_json
        
        '''To pickle and json'''
        # 날짜 O
        with open (f"{weight_path}/weight_log/{save_name}.pkl", 'wb') as f:
            cPickle.dump(meta_data, f)
        with open(f"{weight_path}/weight_log/{save_name}.json", 'w', encoding='utf-8')  as f:
            json.dump(meta_data, f, indent=4, ensure_ascii=False)
        # 날짜 X
        with open(f"{weight_path}/{save_name}.pkl", 'wb') as f:
            cPickle.dump(meta_data, f)
        with open(f"{weight_path}/{save_name}.json", 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=4, ensure_ascii=False)
            
        print(f"Weights and meta data are saved successfully")
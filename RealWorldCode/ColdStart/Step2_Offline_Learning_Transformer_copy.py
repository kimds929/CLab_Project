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

save_file_name_base = f"Real_Alg4_pathtime_var_weights_Transformer_V3"

# system input arguments
gpu_no = 0
num_epochs = 5
batch_size = 64
post_fix = ''

if ('ipykernel_launcher' not in sys.argv[0]) and  len(sys.argv) > 0:
    if len(sys.argv) > 1:
        gpu_no = int(sys.argv[1])

    if len(sys.argv) > 2:
        num_epochs = int(sys.argv[2])
    
    if len(sys.argv) > 3:
        batch_size = int(sys.argv[3])
    
    if len(sys.argv) > 4:
        post_fix = sys.argv[4]
    print(f'(input_arguments) gpu_no: {gpu_no} / num_epochs: {num_epochs} / batch_size: {batch_size}')
save_file_name_base = f"{save_file_name_base}_{post_fix}" if post_fix != '' else f"{save_file_name_base}"
print(save_file_name_base)

############################################################################################################

print(f" *** Load Library *** ")

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
from module.Module_TorchModel import CombinedEmbedding, BaroNowResNet_V3, BaroNowTransformer_V3

from geopy.distance import geodesic
import holidays
import pickle
import copy
##############################################################################################################


print(f" *** Initial Setting *** ")

kst = pytz.timezone('Asia/Seoul')
now_date = datetime.strftime(datetime.now(kst), "%Y-%m-%dT%H:%M:%S%z")
save_file_name = f"{now_date[:10]}_" + save_file_name_base
target_time_safe_margin = 8*60

# meta data --------------------------------------------------------------------------------------------
meta_data_weight = {
        "header": {
            "title": None,
            "description":None,
            "last_update": datetime.strftime(datetime.now(kst), "%Y-%m-%dT%H:%M:%S%z") ,
            "path": None,
            "source": []
        },
        "body": {
            "feature_columns" : {},
            "hyper_params": [],
            "model_structures": [],
            "weights": []
        }
    }

meta_data_weight['header']['title'] = f'{save_file_name}'
meta_data_weight['header']['description'] = "Alg4 PathTime과 Variance 예측을 위한 Model Weight"

##############################################################################################################






# feature_dict['api'] = {}
# feature_dict['api']['token_cols'] = ['transportation']
# feature_dict['api']['temporal_cols'] = ['target_time', 'call_time_LastAPI']
# feature_dict['api']['spatial_cols'] = ['start_point', 'target_point']
# feature_dict['api']['numerical_cols'] =  ['path_time_TimeMachine','path_time_LastAPI']

# batch_idx = next(iter(train_indices_dataloader))[0].numpy()
# batch_data = data_training.loc[batch_idx]

# baronow_preprocess = BaroNowPreprocessing()
# baronow_preprocess.preprocessing(data=batch_data, feature_dict=feature_dict['base'], fillna=0).shape
# baronow_preprocess.preprocessing(data=batch_data, feature_dict=feature_dict['timemachine'], fillna=0)
# baronow_preprocess.preprocessing(data=batch_data, feature_dict=feature_dict['api'], fillna=0).shape

# a2 = baronow_preprocess.preprocessing(data=batch_data, feature_dict=feature_dict['timemachine'], fillna=0)
# ce2 = CombinedEmbedding(numerical_input_dim=1, numerical_emb_dim=3)
# ce2(a2).shape

# a3 = baronow_preprocess.preprocessing(data=batch_data, feature_dict=feature_dict['api'], fillna=0)
# baronow_preprocess.encoding_df
# ce3 = CombinedEmbedding(numerical_input_dim=1, numerical_emb_dim=3, temporal_input_dim=4, temporal_emb_dim=5)
# ce3(a3)


# a1 = baronow_preprocess.preprocessing(data=batch_data, feature_dict=feature_dict['base'], fillna=0)
# baronow_preprocess.encoding_df
# baronow_preprocess.encoding_cols_dict




##############################################################################################################


# dataset_load
print(f" *** Data Load *** ")
data_timemachine = cPickle.load(open(f"{dataset_path}/Alg4_TrainingData_timemachine.pkl", 'rb'))
data_realtime = cPickle.load(open(f"{dataset_path}/Alg4_TrainingData_realtime.pkl", 'rb'))
data_timemachine['optimalAPI_path_time'] = data_timemachine['optimal_delta'] + data_timemachine['path_time_TimeMachine']
print(data_timemachine.shape, data_realtime.shape)

training_source = sorted(data_realtime['id'].drop_duplicates().apply(lambda x: x.split('_')[-1].split('T')[0].replace('-','')).drop_duplicates().to_list())

# train-test split
test_size = 0.2
print(f" *** Train-Test Split *** | test_size : {test_size}")
rng = np.random.RandomState()

# time_machine
indices_timemahcine = rng.permutation(data_timemachine.index)
len_train_timemahcine = int(len(indices_timemahcine) * (1-test_size))
train_indices_timemachine = indices_timemahcine[:len_train_timemahcine]
test_indices_timemachine = indices_timemahcine[len_train_timemahcine:]

data_train_timemachine = data_timemachine.loc[train_indices_timemachine]
data_test_timemachine = data_timemachine.loc[test_indices_timemachine]

print('[timemachine]', data_train_timemachine.shape, data_test_timemachine.shape)

# real_time
indices_timemahcine = rng.permutation(data_realtime.index)
len_train_timemahcine = int(len(indices_timemahcine) * (1-test_size))
train_indices_realtime = indices_timemahcine[:len_train_timemahcine]
test_indices_realtime = indices_timemahcine[len_train_timemahcine:]

data_train_realtime = data_realtime.loc[train_indices_realtime]
data_test_realtime = data_realtime.loc[test_indices_realtime]

print('[real_time]', data_train_realtime.shape, data_test_realtime.shape)


############################################################################################################
# Feature-Set -----------------------------------------------------------------------------
feature_dict = {}
feature_dict['base'] = {}
feature_dict['base']['token_cols'] = ['transportation']
feature_dict['base']['temporal_cols'] = ['target_time']
feature_dict['base']['spatial_cols'] = ['start_point', 'target_point']
feature_dict['base']['numerical_cols'] =  []

feature_dict['timemachine'] = {}
feature_dict['timemachine']['numerical_cols'] = ['path_time_TimeMachine']

feature_dict['api'] ={}
feature_dict['api']['temporal_cols'] = ['call_time_LastAPI']
feature_dict['api']['numerical_cols'] = ['path_time_LastAPI']

baronow_preprocess = BaroNowPreprocessing()
torch_sample1 = baronow_preprocess.preprocessing(data=data_train_timemachine.sample(10), feature_dict=feature_dict['base'])
torch_sample2 = baronow_preprocess.preprocessing(data=data_train_timemachine.sample(10), feature_dict=feature_dict['timemachine'])
torch_sample3 = baronow_preprocess.preprocessing(data=data_realtime.sample(10), feature_dict=feature_dict['api'])
# pd.DataFrame(torch_timemachine.numpy())

############################################################################################################
# Troch Dataset / DataLoader -----------------------------------------------------------------------------
print(f" *** DataSet / DataLoader *** ")
from torch.utils.data import DataLoader, TensorDataset

# # dynamic dataset (base)
# def generate_dyanmic_dataloader(dataset, test_size=0.3, batch_size=64, random_State=None):
#     rng = np.random.RandomState(random_State)
#     sample_indices = rng.choice(dataset.index, size=int(len(dataset)*test_size))

#     data_dynamic_sample = dataset.loc[sample_indices]
#     data_dynamic_sample['start_point'] = data_dynamic_sample['target_point']
#     data_dynamic_sample['optimalAPI_path_time'] = 0
#     data_dynamic = pd.concat([data_dynamic_sample, dataset], axis=0)
#     data_dynamic = data_dynamic.reset_index(drop=True)
#     # print(dataset.shape, data_dynamic_sample.shape, data_dynamic.shape)

#     indices_torch_dynamic = torch.tensor(data_dynamic.index).type(torch.int64)
#     indices_dataset_dynamic = TensorDataset(indices_torch_dynamic)
#     indices_dataloader_dynamic = DataLoader(indices_dataset_dynamic, batch_size=batch_size, shuffle=True)
#     # data_dynamic.loc[next(iter(indices_dataloader_dynamic))[0].numpy()][['start_point','target_point','optimalAPI_path_time']]
#     return (data_dynamic, indices_dataloader_dynamic)

# train_data_dynamic, train_indices_dataloader_dynamic = generate_dyanmic_dataloader(data_train_timemachine, batch_size=batch_size)
# test_data_dynamic, test_indices_dataloader_dynamic = generate_dyanmic_dataloader(data_test_timemachine, batch_size=batch_size)
# print(f'<DataLoader : # of batch> [base] train: {len(train_indices_dataloader_dynamic)} / test: {len(test_indices_dataloader_dynamic)}')


# time_machine
train_indices_torch_timemachine = torch.tensor(data_train_timemachine.index).type(torch.int64)
test_indices_torch_timemachine = torch.tensor(data_test_timemachine.index).type(torch.int64)

train_indices_dataset_timemachine = TensorDataset(train_indices_torch_timemachine)
train_indices_dataloader_timemachine = DataLoader(train_indices_dataset_timemachine, batch_size=batch_size, shuffle=True)

test_indices_dataset_timemachine = TensorDataset(test_indices_torch_timemachine)
test_indices_dataloader_timemachine = DataLoader(test_indices_dataset_timemachine, batch_size=batch_size, shuffle=True)

print(f'<DataLoader : # of batch> [timeamchine] train: {len(train_indices_dataloader_timemachine)} / test: {len(test_indices_dataloader_timemachine)}')


# real_time
train_indices_torch_realtime = torch.tensor(data_train_realtime.index).type(torch.int64)
test_indices_torch_realtime = torch.tensor(data_test_realtime.index).type(torch.int64)

train_indices_dataset_realtime = TensorDataset(train_indices_torch_realtime)
train_indices_dataloader_realtime = DataLoader(train_indices_dataset_realtime, batch_size=batch_size, shuffle=True)

test_indices_dataset_realtime = TensorDataset(test_indices_torch_realtime)
test_indices_dataloader_realtime = DataLoader(test_indices_dataset_realtime, batch_size=batch_size, shuffle=True)

print(f'<DataLoader : # of batch> [realtime] train: {len(train_indices_dataloader_realtime)} / test: {len(test_indices_dataloader_realtime)}')

# # samples
# batch_idx = next(iter(train_indices_dataloader_realtime))[0].numpy()
# batch_data = data_train_realtime.loc[batch_idx]

# batch_torch_X1 = baronow_preprocess.preprocessing(data=batch_data, feature_dict=feature_dict['base'])
# batch_torch_X2 = baronow_preprocess.preprocessing(data=batch_data, feature_dict=feature_dict['timemachine'])
# batch_torch_X3 = baronow_preprocess.preprocessing(data=batch_data, feature_dict=feature_dict['api'])
# batch_torch_y = torch.tensor(batch_data['optimal_delta'].to_numpy().reshape(-1,1)).type(torch.float32)
# print(batch_torch_X1.shape, batch_torch_X2.shape, batch_torch_X3.shape, batch_torch_y.shape)



############################################################################################################
print(f" *** Device / Model *** ")
import torch.nn as nn
import torch.optim as optim

# device
device = torch.device(f'cuda:{gpu_no}') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# hyper-parameter -------------------------------------------------------------
base_token_dim = 1
base_temporal_dim = 4

timemachine_numerical_dim = 1
api_numerical_dim = 1
api_temporal_dim = 4

hyper_parmas_dict = {}
hyper_parmas_dict['base_embedding_params'] = {}
hyper_parmas_dict['base_embedding_params']["categorical_num_embedding"] = 5
hyper_parmas_dict['base_embedding_params']["categorical_input_dim"] = base_token_dim
hyper_parmas_dict['base_embedding_params']["categorical_emb_dim"] = base_token_dim*8
hyper_parmas_dict['base_embedding_params']["temporal_input_dim"] = base_temporal_dim
hyper_parmas_dict['base_embedding_params']["temporal_emb_dim"] = base_temporal_dim*16
hyper_parmas_dict['base_embedding_params']["temporal_params"] = {"hidden_dim":base_temporal_dim*32}
hyper_parmas_dict['base_embedding_params']["spatial_emb_dim"] = 4*8

hyper_parmas_dict['timemachine_embedding_params'] = {}
hyper_parmas_dict['timemachine_embedding_params']["numerical_input_dim"] = timemachine_numerical_dim
hyper_parmas_dict['timemachine_embedding_params']["numerical_emb_dim"] = timemachine_numerical_dim*8

hyper_parmas_dict['api_embedding_params'] = {}
hyper_parmas_dict['api_embedding_params']["numerical_input_dim"] = api_numerical_dim
hyper_parmas_dict['api_embedding_params']["numerical_emb_dim"] = api_numerical_dim*8
hyper_parmas_dict['api_embedding_params']["temporal_input_dim"] = api_temporal_dim
hyper_parmas_dict['api_embedding_params']["temporal_emb_dim"] = api_temporal_dim*16
hyper_parmas_dict['api_embedding_params']["temporal_params"] = {"hidden_dim":api_temporal_dim*32}
# -----------------------------------------------------------------------------


# model define -----------------------------------------------------------------------------
# from module.Module_TorchModel import BaroNowModel_V2, BaroNowTransformer_V3

model = BaroNowTransformer_V3(**hyper_parmas_dict, n_output=10)
print("# of params in model : ", sum(p.numel() for p in model.parameters() if p.requires_grad))


# batch_torch_y
# model(batch_torch_X1, pred_type='std')
# model(batch_torch_X1, batch_torch_X2, pred_type='std')
# model(batch_torch_X1, batch_torch_X2, batch_torch_X3, pred_type='std')
# pd.Series(model.requires_grad_status()).to_frame().to_csv(f"{base_path}/weight_require_grad.csv")


# # load model weights ----------------------------------------------------------------------
# # (case 1) load from pkl
# if f"{save_file_name_base}.pkl" in os.listdir(f"{weight_path}"):
#     state_dict = cPickle.load(open(f"{weight_path}/{save_file_name_base}.pkl", 'rb'))
#     model.load_state_dict(state_dict)


# (case 2) load from json
if f"{save_file_name_base}.json" in os.listdir(f"{weight_path}"):
    with open(f"{weight_path}/{save_file_name_base}.json", "r") as file:
        meta_data_weight = json.load(file)
    meta_data_weight['header']
    state_dict = list_to_tensor(meta_data_weight['body']['weights'])
    model.load_state_dict(state_dict)
    print('★★★ weight load complete. ★★★')
# # ----------------------------------------------------------------------------------------------

model.to(device)    # model_to_device

# -------------------------------------------------------------------------------------------


# (Offline Leaning) -----------------------------------------------------------------------------------------------
print(f" *** Training *** ")

optimizer = optim.Adam(model.parameters(), lr=1e-5)
# optimizer = optim.AdamW(model_pathtime.parameters(), lr=1e-5, weight_decay=1e-2)
# optimizer = optim.RMSprop(model_pathtime.parameters(), lr=1e-5)
# scheduler_var = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# scheduler_var = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)



def run_process(model, dataframe, index_loader, feature_dicts={}, pred_label='optimal_delta', is_training=True, device='cpu', prefix=None):
    label = 'Train'if is_training else 'Test'
    prefix = f'_{prefix}' if prefix is not None else ''

    batch_gaussian_loss = []
    batch_mse_loss = []
    batch_delta_mean = []
    batch_delta_std = []

    with torch.no_grad() if not is_training else torch.enable_grad():
        with tqdm(total=len(index_loader), desc=f"({label}{prefix}) Epoch {epoch+1}/{num_epochs}") as pbar:
            model.train() if is_training else model.eval()

            for bi, (batch_idx) in enumerate(index_loader):
                batch_idx = batch_idx[0].numpy()

                # batch_data
                batch_data = dataframe.loc[batch_idx]
                batch_x_dict = {}
                for arg_name, feature_dict in feature_dicts.items():
                    batch_x_dict[arg_name] = baronow_preprocess.preprocessing(data=batch_data, feature_dict=feature_dict).to(device)
                batch_y = (torch.tensor(batch_data[pred_label].to_numpy().reshape(-1,1)).type(torch.float32)/60).to(device)

                # training_mu
                pred_mu = model(**batch_x_dict, pred_type='mu')
                mse_loss = nn.functional.mse_loss(pred_mu, batch_y)
                if is_training is True:
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()
                
                # training_std
                pred_std = model(**batch_x_dict, pred_type='std')
                pred_mu_ = pred_mu.detach()
                gaussian_loss = nn.functional.gaussian_nll_loss(pred_mu_, batch_y, pred_std**2)
                if is_training is True:
                    optimizer.zero_grad()
                    gaussian_loss.backward()
                    optimizer.step()

                # save learning history
                with torch.no_grad():

                    batch_gaussian_loss.append( gaussian_loss.item() )
                    batch_mse_loss.append( mse_loss.item() )
                    batch_delta_mean.append( torch.mean(pred_mu).item() )
                    batch_delta_std.append( torch.mean(pred_std).item() )
                
                # tqdm display
                if bi % 1 == 0:
                    pbar.set_postfix(Gaussian_Loss=f"{np.mean(batch_gaussian_loss).item():.2f}", RMSE_Loss=f"{np.sqrt(np.mean(batch_mse_loss)).item():.2f}",
                                     Mean=f"{np.mean(batch_delta_mean).item():.2f}", Std=f"{np.mean(batch_delta_std).item():.2f}")
                pbar.update(1)
                # break
    return {'batch_gaussian_loss': batch_gaussian_loss, 'batch_mse_loss': batch_mse_loss, 
            'batch_delta_mean':batch_delta_mean, 'batch_delta_std': batch_delta_std}

# # ----------------------------------------------------------------------------------------------
# learning_log data load
if f"{save_file_name_base}_learning_log.json" in os.listdir(f"{weight_path}/learning_log"):
    with open(f"{weight_path}/learning_log/{save_file_name_base}_learning_log.json", 'r') as file:
        learning_log = json.load(file) 
    print('★★★ log load complete. ★★★')
else:
    # learning log
    learning_log = {}
    learning_log['TrainBase'] = []
    learning_log['TrainTimeMachine'] = []
    learning_log['TrainAPI'] = []
    learning_log['TestBase'] = []
    learning_log['TestTimeMachine'] = []
    learning_log['TestAPI'] = []
# # ----------------------------------------------------------------------------------------------

# pre_training of standard deviation for fast convergence
# num_pre_epoch = 50
# for pre_epoch in range(num_pre_epoch):
#     pre_mu = []
#     pre_std = []
#     pre_mu_loss = []
#     pre_std_loss = []
#     with tqdm(total=len(train_indices_dataloader_timemachine), desc=f"Epoch {pre_epoch+1}/{num_pre_epoch}") as pbar:
#         for bi, (batch_idx) in enumerate(train_indices_dataloader_timemachine):
#             batch_idx = batch_idx[0].numpy()
#             # batch_data
#             batch_data = data_train_timemachine.loc[batch_idx]
            
#             batch_X = baronow_preprocess.preprocessing(data=batch_data, feature_dict=feature_dict['base']).to(device)
#             batch_y_mu = torch.tensor(batch_data['optimalAPI_path_time'].to_numpy()/60).view(-1,1).type(torch.float32).to(device)
#             batch_y_std_init = (torch.ones(len(batch_X)).view(-1,1) * (3 * batch_data['optimalAPI_path_time']/60).std()).type(torch.float32).to(device)

#             batch_y = (torch.ones(len(batch_X)).view(-1,1) * (3 * batch_data['optimalAPI_path_time']/60).std()).type(torch.float32).to(device)

#             pred_mu = model(batch_X, pred_type='mu')
#             mu_mse_loss = nn.functional.mse_loss(pred_mu, batch_y_mu)
#             optimizer.zero_grad()
#             mu_mse_loss.backward()
#             optimizer.step()
            
#             # pred_std = model(batch_X, pred_type='std')
#             # std_mse_loss = nn.functional.mse_loss(pred_std, batch_y_std_init)
#             # optimizer.zero_grad()
#             # std_mse_loss.backward()
#             # optimizer.step()

#             pre_mu.append(torch.mean(pred_mu).item())
#             # pre_std.append(torch.mean(pred_std).item())
#             pre_mu_loss.append(mu_mse_loss.item())
#             # pre_std_loss.append(std_mse_loss.item())
#             pbar.set_postfix(Mu=f"{np.mean(pre_mu).item():.2f}" , Std=f"{np.mean(pre_std).item():.2f}",
#                             Mu_MSE_Loss = f"{np.mean(pre_mu_loss).item():.2f}"
#                             # Std_MSE_Loss = f"{np.mean(pre_std_loss).item():.2f}"
#                             )
#             pbar.update(1)



# num_epochs = 3
for epoch in range(num_epochs):
    learning_result = {}
    
    print(f'###### Epoch {epoch+1}/{num_epochs} ################################################################## ')
    # (training) -------------------------------------------------------------------------------------------------
    # base prediction
    for _ in range(3):
        batch_result = run_process(model=model, dataframe=data_train_timemachine, index_loader=train_indices_dataloader_timemachine,
                                feature_dicts={'base_x':feature_dict['base']},
                                pred_label='optimalAPI_path_time', is_training=True, device=device, prefix='BASE')
        # train_data_dynamic, train_indices_dataloader_dynamic = generate_dyanmic_dataloader(data_train_timemachine, batch_size=batch_size)
        # batch_result = run_process(model=model, dataframe=train_data_dynamic, index_loader=train_indices_dataloader_dynamic,
        #                         feature_dicts={'base_x':feature_dict['base']}, 
        #                         pred_label='optimalAPI_path_time', is_training=True, device=device, prefix='BASE')
        learning_log['TrainBase'].append(batch_result)

    # timemachine prediction
    batch_result = run_process(model=model, dataframe=data_train_timemachine, index_loader=train_indices_dataloader_timemachine,
                            feature_dicts={'base_x':feature_dict['base'], 'timemachine_x': feature_dict['timemachine']},
                            pred_label='optimal_delta', is_training=True, device=device, prefix='TM')
    learning_log['TrainTimeMachine'].append(batch_result)

    # api prediction
    batch_result = run_process(model=model, dataframe=data_train_realtime, index_loader=train_indices_dataloader_realtime,
                            feature_dicts={'base_x':feature_dict['base'], 'timemachine_x': feature_dict['timemachine'], 'api_x':feature_dict['api']},
                            pred_label='optimal_delta', is_training=True, device=device, prefix='API')
    learning_log['TrainAPI'].append(batch_result)
    print()

    # (test) -------------------------------------------------------------------------------------------------
    # base prediction
    batch_result = run_process(model=model, dataframe=data_test_timemachine, index_loader=test_indices_dataloader_timemachine,
                            feature_dicts={'base_x':feature_dict['base']},
                            pred_label='optimalAPI_path_time', is_training=False, device=device, prefix='BASE')
    # test_data_dynamic, test_indices_dataloader_dynamic = generate_dyanmic_dataloader(data_test_timemachine, batch_size=batch_size)
    # batch_result = run_process(model=model, dataframe=test_data_dynamic, index_loader=test_indices_dataloader_dynamic,
    #                         feature_dicts={'base_x':feature_dict['base']}, 
    #                         pred_label='optimalAPI_path_time', is_training=False, device=device, prefix='BASE')
    learning_log['TestBase'].append(batch_result)

    # timemachine prediction
    batch_result = run_process(model=model, dataframe=data_test_timemachine, index_loader=test_indices_dataloader_timemachine,
                            feature_dicts={'base_x':feature_dict['base'], 'timemachine_x': feature_dict['timemachine']},
                            pred_label='optimal_delta', is_training=False, device=device, prefix='TM')
    learning_log['TestTimeMachine'].append(batch_result)

    # api prediction
    batch_result = run_process(model=model, dataframe=data_test_realtime, index_loader=test_indices_dataloader_realtime,
                            feature_dicts={'base_x':feature_dict['base'], 'timemachine_x': feature_dict['timemachine'], 'api_x':feature_dict['api']},
                            pred_label='optimal_delta', is_training=False, device=device, prefix='API')
    learning_log['TestAPI'].append(batch_result)
    print()


    # save weights -------------------------------------------------------------------------------
    print(f" *** Weight Save *** ")

    state_dict = model.state_dict()
    state_dict_json = tensor_to_list(state_dict)
    cPickle.dump(state_dict, open(f"{weight_path}/{save_file_name}.pkl", 'wb'))
    cPickle.dump(state_dict, open(f"{weight_path}/{save_file_name_base}.pkl", 'wb'))

    meta_data_weight['header']['last_update'] = now_date
    meta_data_weight['header']['path'] = f"{weight_path}/{save_file_name}.pkl"
    meta_data_weight['header']['source'] = training_source
    meta_data_weight['body']['hyper_params'] = hyper_parmas_dict
    meta_data_weight['body']['model_structures'] = str(model)
    meta_data_weight['body']['weights'] = state_dict_json
    meta_data_weight['body']['feature_columns'] = feature_dict

    # save info to json
    with open(f"{weight_path}/{save_file_name}.json", 'w', encoding='utf-8') as file:
        json.dump(meta_data_weight, file, indent=4, ensure_ascii=False) 
    with open(f"{weight_path}/{save_file_name_base}.json", 'w', encoding='utf-8') as file:
        json.dump(meta_data_weight, file, indent=4, ensure_ascii=False) 
    print('save complete.') 

    # save history
    with open(f"{weight_path}/learning_log/{save_file_name_base}_learning_log.json", 'w', encoding='utf-8') as file:
        json.dump(learning_log, file, indent=4, ensure_ascii=False) 
    # ---------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------
print(f" *** Complete Step1_Offline_Learning.py *** ")



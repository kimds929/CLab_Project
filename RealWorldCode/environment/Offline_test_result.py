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


offline_test_result_folder_path = f"{env_path}/offline_test_result"
# os.listdir(f"{offline_test_result_folder_path}")
# file_name = '2024-12-18T11:48:41+0900_resnet_alpha4.0_gamma2.5_offline_test_result.csv'

verbose = 0

file_name.split('_')
test_results = []
# file_name = 'all_2024-12-20T05:56:47+0900_Rule_alpha3_gammaNone_max_api_call1_offline_test_result.csv'
for file_name in os.listdir(f"{offline_test_result_folder_path}"):
    if 'csv' in file_name:
        test_type, date, model, alpha, gamma, _, _, max_api_call = file_name.split('_')[:8]
        interval = int(file_name.split('_')[10].replace('interval',''))
        name = '_'.join(file_name.split('_')[1:4])
        df_temp = pd.read_csv(f"{offline_test_result_folder_path}/{file_name}", encoding='utf-8-sig')
        n_episodes = len(df_temp)

        # elements
        accept_ratio = df_temp['arrival'].sum() / len(df_temp)
        excess_ratio = df_temp[(df_temp['result'] > 0)].shape[0] / len(df_temp)
        mean_n_api_call = df_temp['n_of_api_call'].mean()
        total_mean = df_temp['result'].mean()
        total_std = df_temp['result'].std()
        excess_mean = df_temp[df_temp['result'] > 0]['result'].mean()
        early_mean = df_temp[df_temp['result'] < -13]['result'].mean()

        if verbose > 0:
            print(f'< summary > {name}')
            print(f"(n_episodes : {n_episodes}) accept_ratio: {accept_ratio: .3f}, excess_ratio: {excess_ratio:.3f}, mean_n_api_call: {mean_n_api_call:.2f}")
        
        alpha_save = float(alpha.replace('alpha','')) if alpha != 'alphaNone' else None
        gamma_save = float(gamma.replace('gamma','')) if gamma != 'gammaNone' else None

        file_info = {}
        # file_info['name'] = name
        file_info['date'] = date
        file_info['test_type'] = test_type
        file_info['model'] = model
        file_info['n_episodes'] = n_episodes
        file_info['alpha'] = alpha_save
        file_info['gamma'] = gamma_save
        file_info['min_api_interval'] = interval
        file_info['max_api_call'] = max_api_call.replace('call','').replace('.csv','')
        file_info['accept_ratio'] = accept_ratio
        file_info['excess_ratio'] = excess_ratio
        file_info['early_ratio'] = 1-accept_ratio-excess_ratio
        file_info['mean_n_api_call'] = mean_n_api_call
        file_info['total_mean'] = total_mean
        file_info['total_std'] = total_std
        file_info['excess_mean'] = excess_mean
        file_info['early_mean'] = early_mean
        test_results.append(file_info)

df_result = pd.DataFrame(test_results)
df_result['model'] = pd.Categorical(df_result['model'], categories=['Rule', 'Alg3', 'resnet', 'transformer'], ordered=True)
df_result = df_result.sort_values(['test_type','max_api_call', 'model', 'alpha', 'gamma', 'min_api_interval'], axis=0, ascending=[True, True, True, True, True, True])

# df_result['before_ratio'] = df_result['before_ratio'].apply(lambda x: round(x,6))
for col in ['accept_ratio', 'excess_ratio', 'early_ratio', 'mean_n_api_call', 'total_mean', 'total_std', 'excess_mean', 'early_mean']:
    df_result[col] = df_result[col].apply(lambda x: np.round(x,3))

df_result

df_result = df_result.set_index('date')
df_result.sort_index().iloc[-8:]


# df_result.to_csv(f"{env_path}/offline_test_result.csv", index=False, encoding='utf-8-sig')


# for file_name in os.listdir(f"{offline_test_result_folder_path}"):
#     if 'min_api_interval' not in file_name:
#         print(file_name)
#         old_name_path = f"{env_path}/offline_test_result/{file_name}"
#         new_name = '_'.join(file_name.split('_')[:-3] + ['min', 'api', 'interval3','offline','test','result.csv'])
#         new_name_path = f"{env_path}/offline_test_result/{new_name}"
#         os.rename(old_name_path, new_name_path)




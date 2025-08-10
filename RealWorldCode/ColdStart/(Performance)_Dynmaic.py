
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
from geopy.distance import geodesic
##############################################################################################################

from module.Module_Preprocessing import datetime_encoding, BaroNowPreprocessing
from module.Module_Utils import tensor_to_list, list_to_tensor
from module.Module_TorchModel import BaroNowResNet_V3, BaroNowTransformer_V3, BaroNowResNet_V2, BaroNowTransformer_V2

# ---------------------------------------------------------
df_optimal = pd.read_csv(f"{dataset_path}/Optimal_API_Call_DataSet.csv", encoding='utf-8-sig')
print(df_optimal.shape)
# ---------------------------------------------------------


with open(f"{weight_path}/Real_Alg4_pathtime_var_weights_ResNet_V2.json", 'r') as file:
    meta_data_resnet_v2 = json.load(file)

with open(f"{weight_path}/Real_Alg4_pathtime_var_weights_Transformer_V2.json", 'r') as file:
    meta_data_transformer_v2 = json.load(file)

with open(f"{weight_path}/Real_Alg4_pathtime_var_weights_ResNet_V3.json", 'r') as file:
    meta_data_resnet_v3 = json.load(file)

with open(f"{weight_path}/Real_Alg4_pathtime_var_weights_Transformer_V3.json", 'r') as file:
    meta_data_transformer_v3 = json.load(file)

with open(f"{weight_path}/Real_Alg4_pathtime_var_weights_Transformer_V3_pretrain.json", 'r') as file:
    meta_data_transformer_v3_pretrain = json.load(file)


model = BaroNowResNet_V2(**meta_data_resnet_v2['body']['hyper_params'])
model.load_state_dict(list_to_tensor(meta_data_resnet_v2['body']['weights']))
feature_dict_base = meta_data_resnet_v2['body']['feature_columns']['base']

# model = BaroNowTransformer_V2(**meta_data_resnet_v2['body']['hyper_params'])
# model.load_state_dict(list_to_tensor(meta_data_transformer_v2['body']['weights']))
# feature_dict_base = meta_data_transformer_v2['body']['feature_columns']['base']

# model = BaroNowResNet_V3(**meta_data_resnet_v3['body']['hyper_params'])
# model.load_state_dict(list_to_tensor(meta_data_resnet_v3['body']['weights']))
# feature_dict_base = meta_data_resnet_v3['body']['feature_columns']['base']

# model = BaroNowTransformer_V3(**meta_data_transformer_v3['body']['hyper_params'])
# model.load_state_dict(list_to_tensor(meta_data_transformer_v3['body']['weights']))
# feature_dict_base = meta_data_transformer_v3['body']['feature_columns']['base']

# model = BaroNowTransformer_V3(**meta_data_transformer_v3_pretrain['body']['hyper_params'])
# model.load_state_dict(list_to_tensor(meta_data_transformer_v3_pretrain['body']['weights']))
# feature_dict_base = meta_data_transformer_v3_pretrain['body']['feature_columns']['base']



size = 300
preprocess_instant = BaroNowPreprocessing()
df_type = preprocess_instant.type_transform(df_optimal.sample(size))
df_encode, _ = preprocess_instant.encoding(df_type, feature_dict_base)
# preprocess_instant.to_torch(df_encode)
# preprocess_instant.preprocessing(df_optimal.sample(size), feature_dict_base)

# df_enc = preprocess_instant.encoding_df
df_enc = df_encode.copy()

def calc_dist(coord_1x, coord_1y, coord_2x, coord_2y):
    x1 = (coord_1x * (43-33)) + 33
    y1 = (coord_1y * (132-124)) + 124
    x2 = (coord_2x * (43-33)) + 33
    y2 = (coord_2y * (132-124)) + 124
    dist = geodesic((x1, y1), (x2, y2)).kilometers
    return dist

df_enc['dist'] = df_enc.apply(lambda x: calc_dist(x['start_point_x'], x['start_point_y'], x['target_point_x'], x['target_point_y']), axis=1)

# rng = np.random.RandomState()
# x_samples = rng.normal(loc=0.451786, scale=0.014266, size=size*2)
# y_samples = rng.normal(loc=0.451786, scale=0.014266, size=size*2)

# x_samples_coord = (x_samples * (43-33)) + 33
# y_samples_coord = (y_samples * (132-124)) + 124

# start_point_x = x_samples_coord[:size]
# start_point_y = y_samples_coord[:size]
# target_point_x = x_samples_coord[size:]
# target_point_y = y_samples_coord[size:]

# coords = np.stack([start_point_x, start_point_y, target_point_x, target_point_y]).T

# result = []
# for coord in coords:
#     sp = coord[0].item(), coord[1].item()
#     tp = coord[2].item(), coord[3].item()

#     dist = geodesic(sp, tp).kilometers
#     result.append(dist)


# coords_norm = np.stack([x_samples[:size], y_samples[:size], x_samples[size:], y_samples[size:]]).T
# coord_data = pd.DataFrame(coords_norm, columns=['start_point_x', 'start_point_y', 'target_point_x', 'target_point_y'])
# coord_data['dist'] = result
# coord_data = coord_data.sort_values('dist', axis=0)
# coord_data.index = df_enc.index


# df_enc['start_point_x'] = coord_data['start_point_x']
# df_enc['start_point_y'] = coord_data['start_point_y']
# df_enc['target_point_x'] = coord_data['target_point_x']
# df_enc['target_point_x'] = coord_data['target_point_x']

# coord_data
# df_enc.iloc[:,-4:]
# coord_data['dist']

# df_type.loc[df_enc.index]['path_time_LastAPI']
df_enc = df_enc.sort_values('dist', axis=0)
df_torch = preprocess_instant.to_torch(df_enc.iloc[:,:-1])



pred_mu = model(df_torch)[0].view(-1).detach()
pred_std = model(df_torch)[1].view(-1).detach()

plt.figure(figsize=(20,15))
plt.plot(df_enc['dist'].to_list(), pred_mu, 'o-', alpha=0.5, label='pred')
plt.fill_between(df_enc['dist'].to_list(), pred_mu-1*pred_std, pred_mu+1*pred_std, alpha=0.3)
plt.plot(df_enc['dist'].to_list(), df_type.loc[df_enc.index]['path_time_LastAPI']/60, 'o-', alpha=0.5, color='orange', label='true')
plt.xlabel('distance')
plt.ylabel('path_time')
plt.ylim(-5, 200)
plt.legend()
# plt.plot(df_enc['dist'].to_list(), pred_std, 'o-', alpha=0.5)
plt.show()



# plt.scatter(df_type.loc[df_enc.index]['path_time_LastAPI']/60, pred_mu)


















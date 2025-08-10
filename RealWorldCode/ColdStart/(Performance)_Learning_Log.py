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

cold_start_path = os.path.join(base_path, 'ColdStart');  sys.path.append( cold_start_path )
dataset_path = os.path.join(base_path, 'dataset');  sys.path.append( dataset_path )
env_path = os.path.join(base_path, 'environment');  sys.path.append( env_path )
logging_path = os.path.join(base_path, 'logging');  sys.path.append( logging_path )
module_path = os.path.join(base_path, 'module');    sys.path.append( module_path )
model_path = os.path.join(base_path, 'model');  sys.path.append( model_path )
weight_path = os.path.join(base_path, 'weight');    sys.path.append( weight_path )
############################################################################################################

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


with open(f"{weight_path}/learning_log/Real_Alg4_pathtime_var_weights_ResNet_V2_learning_log.json", 'r') as file:
    learning_log_resnet_v2 = json.load(file)

with open(f"{weight_path}/learning_log/Real_Alg4_pathtime_var_weights_Transformer_V2_learning_log.json", 'r') as file:
    learning_log_transformer_v2 = json.load(file)

with open(f"{weight_path}/learning_log/Real_Alg4_pathtime_var_weights_ResNet_V3_learning_log.json", 'r') as file:
    learning_log_resnet_v3 = json.load(file)

with open(f"{weight_path}/learning_log/Real_Alg4_pathtime_var_weights_Transformer_V3_learning_log.json", 'r') as file:
    learning_log_transformer_v3 = json.load(file)

with open(f"{weight_path}/learning_log/Real_Alg4_pathtime_var_weights_Transformer_V3_pretrain_learning_log.json", 'r') as file:
    learning_log_transformer_v3_pretrain = json.load(file)

# -----------------------------------------------------------------------




# learning_log = learning_log_resnet_v2.copy()
# learning_log = learning_log_transformer_v2.copy()

# learning_log = learning_log_resnet_v3.copy()
# learning_log = learning_log_transformer_v3.copy()
# learning_log = learning_log_transformer_v3_pretrain.copy()

model_name = 'transformer_v3_pretrain'
learning_log = eval(f"learning_log_{model_name}")

# -----------------------------------------------------------------------

# learning_log.keys()
train1 = pd.DataFrame(learning_log['TrainBase']).applymap(lambda x: np.mean(x))
train2 = pd.DataFrame(learning_log['TrainTimeMachine']).applymap(lambda x: np.mean(x))
train3 = pd.DataFrame(learning_log['TrainAPI']).applymap(lambda x: np.mean(x))
train1.columns = [f"train_base_{c}" for c in train1.columns]
train2.columns = [f"train_tm_{c}" for c in train2.columns]
train3.columns = [f"train_api_{c}" for c in train3.columns]

test1 = pd.DataFrame(learning_log['TestBase']).applymap(lambda x: np.mean(x))
test2 = pd.DataFrame(learning_log['TestTimeMachine']).applymap(lambda x: np.mean(x))
test3 = pd.DataFrame(learning_log['TestAPI']).applymap(lambda x: np.mean(x))
test1.columns = [f"test_base_{c}" for c in test1.columns]
test2.columns = [f"test_tm_{c}" for c in test2.columns]
test3.columns = [f"test_api_{c}" for c in test3.columns]


train_log_df = pd.concat([train1, train2, train3], axis=1)
test_log_df = pd.concat([test1, test2, test3], axis=1)

train_log_df.to_csv(f"{cold_start_path}/training_log/performance_train_{model_name}.csv", index=False, encoding='utf-8-sig')
test_log_df.to_csv(f"{cold_start_path}/training_log/performance_test_{model_name}.csv", index=False, encoding='utf-8-sig')


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

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

learning_log_path = f"{weight_path}/learning_log"

with open(f"{learning_log_path}/Real_Alg4_pathtime_var_weights_ResNet_V2_learning_log.json", "r") as file:
    resnet_V2_learning_log = json.load(file)

with open(f"{learning_log_path}/Real_Alg4_pathtime_var_weights_Transformer_V2_learning_log.json", "r") as file:
    transformer_V2_learning_log = json.load(file)

with open(f"{learning_log_path}/Real_Alg4_pathtime_var_weights_ResNet_V3_learning_log.json", "r") as file:
    resnet_V3_learning_log = json.load(file)

with open(f"{learning_log_path}/Real_Alg4_pathtime_var_weights_Transformer_V3_learning_log.json", "r") as file:
    transformer_V3_learning_log = json.load(file)


model_name = 'transformer_V3'
learning_log = eval(f"{model_name}_learning_log")

# learning_log.keys()
train_df_base = pd.DataFrame(learning_log['TrainBase']).applymap(lambda x: np.mean(x))
train_df_tm = pd.DataFrame(learning_log['TrainTimeMachine']).applymap(lambda x: np.mean(x))
train_df_api= pd.DataFrame(learning_log['TrainAPI']).applymap(lambda x: np.mean(x))
train_df_base.columns = [f"base_{c}" for c in train_df_base.columns]
train_df_tm.columns = [f"tm_{c}" for c in train_df_tm.columns]
train_df_api.columns = [f"api_{c}" for c in train_df_api.columns]

test_df_base = pd.DataFrame(learning_log['TestBase']).applymap(lambda x: np.mean(x))
test_df_tm   = pd.DataFrame(learning_log['TestTimeMachine']).applymap(lambda x: np.mean(x))
test_df_api  = pd.DataFrame(learning_log['TestAPI']).applymap(lambda x: np.mean(x))
test_df_base.columns = [f"base_{c}" for c in test_df_base.columns]
test_df_tm.columns = [f"tm_{c}" for c in test_df_tm.columns]
test_df_api.columns = [f"api_{c}" for c in test_df_api.columns]

train_df = pd.concat([train_df_base, train_df_tm, train_df_api], axis=1)
test_df = pd.concat([test_df_base, train_df_tm, test_df_api], axis=1)

train_df.to_csv(f"{learning_log_path}/peformance_train_{model_name}.csv")
train_df.to_csv(f"{learning_log_path}/peformance_test_{model_name}.csv")


plt.plot(train_df['base_batch_gaussian_loss'], 'o-')
plt.plot(train_df['base_batch_mse_loss'], 'o-')

plt.plot(train_df['api_batch_mse_loss'], 'o-')
plt.plot(train_df['tm_batch_mse_loss'], 'o-')



############################################################################




# model_name = 'transformer_V3'
model_names = ['resnet_V2', 'transformer_V2', 'resnet_V3', 'transformer_V3']
for model_name in model_names:
    learning_log = eval(f"{model_name}_learning_log")

    if model_name != 'transformer_V3':

        # learning_log.keys()
        train_df_base = pd.DataFrame(learning_log['TrainBase']).applymap(lambda x: np.mean(x))
        train_df_tm = pd.DataFrame(learning_log['TrainTimeMachine']).applymap(lambda x: np.mean(x))
        train_df_api= pd.DataFrame(learning_log['TrainAPI']).applymap(lambda x: np.mean(x))
        train_df_base.columns = [f"base_{c}" for c in train_df_base.columns]
        train_df_tm.columns = [f"tm_{c}" for c in train_df_tm.columns]
        train_df_api.columns = [f"api_{c}" for c in train_df_api.columns]

        test_df_base = pd.DataFrame(learning_log['TestBase']).applymap(lambda x: np.mean(x))
        test_df_tm   = pd.DataFrame(learning_log['TestTimeMachine']).applymap(lambda x: np.mean(x))
        test_df_api  = pd.DataFrame(learning_log['TestAPI']).applymap(lambda x: np.mean(x))
        test_df_base.columns = [f"base_{c}" for c in test_df_base.columns]
        test_df_tm.columns = [f"tm_{c}" for c in test_df_tm.columns]
        test_df_api.columns = [f"api_{c}" for c in test_df_api.columns]

        train_df = pd.concat([train_df_base, train_df_tm, train_df_api], axis=1)
        test_df = pd.concat([test_df_base, train_df_tm, test_df_api], axis=1)

        # save to csv
        train_df.to_csv(f"{learning_log_path}/performance_train_{model_name}.csv", index=False, encoding='utf-8-sig')
        test_df.to_csv(f"{learning_log_path}/performance_test_{model_name}.csv", index=False, encoding='utf-8-sig')



model_names = ['resnet_V2', 'transformer_V2', 'resnet_V3', 'transformer_V3']
train_result_list = []
test_result_list = []
for model_name in model_names:
    train_df = pd.read_csv(f"{learning_log_path}/performance_train_{model_name}.csv", encoding='utf-8-sig')
    test_df = pd.read_csv(f"{learning_log_path}/performance_test_{model_name}.csv", encoding='utf-8-sig')

    # save to dict
    train_dict = {}
    train_dict['model_name'] = model_name
    train_dict['type'] = 'train'
    train_dict['n_epochs'] = len(train_df)
    train_dict.update(train_df.iloc[-1].to_dict())
    train_result_list.append(train_dict)

    test_dict = {}
    test_dict['model_name'] = model_name
    test_dict['type'] = 'test'
    test_dict['n_epochs'] = len(test_df)
    test_dict.update(test_df.iloc[-1].to_dict())
    test_result_list.append(test_dict)

df_result_summary = pd.DataFrame(train_result_list + test_result_list)
df_result_summary['model_name'] = pd.Categorical(df_result_summary['model_name'], categories=model_names, ordered=True)

df_result_summary.sort_values(['model_name'], axis=0).to_csv(f"{learning_log_path}/performance_summary.csv", index=False, encoding='utf-8-sig')

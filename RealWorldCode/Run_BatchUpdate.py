############################################################################################################
# (Path Setting)
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

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

from datetime import datetime, timedelta
import pytz
import pandas as pd
# from functools import reduce
from Run_Logging import readAllLog, readLog, episodeFilter, lastTwo, lastTwoContexts, logSave
from model.Algorithm4_Model import BaroNow_Algorithm4_Model as BaroNowModel
from model.Algorithm4_BatchedUpdate import BatchUpdateModel4

kst = pytz.timezone('Asia/Seoul')
today = datetime.now()
today_str = today.strftime('%y%m%d')

csv_path = os.path.join(logging_path, 'batch_update_data.csv')

'''Model & Algorithm'''
baronow_model = BaroNowModel()
model = 'resnet' # or 'transformer'
batch_update_model = BatchUpdateModel4(model)
algorithm = 'BaroNow_Algorithm4_Model'

# 폴더 안에 csv 파일이 존재하는 경우
try:
    existing_data = pd.read_csv(csv_path)
    print(f"Loaded existing batch update data.")
    
    df = readLog(algorithm, today_str)
    df_over_two_line = episodeFilter(df)
    last_two_lines = lastTwo(df_over_two_line)
    df_contexts = lastTwoContexts(last_two_lines)
    
    if not df_contexts.empty:
        today_and_past = pd.concat([existing_data, df_contexts], axis=0)
        today_and_past.to_csv(csv_path, index=False)
        print("Appended today log data to csv.")
    else:
        print("No new logs found for today.")
        
# 폴더 안에 csv 파일이 존재하지 않는 경우 (cold start)
except FileNotFoundError: 
    print("No existing data file found. Creating a new one.")
    
    df = readAllLog() # 폴더 안의 모든 log 파일 읽기
    df_over_two_line = episodeFilter(df) 
    last_two_lines = lastTwo(df_over_two_line)
    df_contexts = lastTwoContexts(last_two_lines)

    if not df_contexts.empty:
        df_contexts.to_csv(csv_path, index=False)
    else:
        print("No log data found to initialize.\n")

# 여기까지 실행하면 오늘 로그까지 csv 파일에 추가됨(만약 정상적인 에피소드가 없다면 아무런 csv 파일도 생기지 않음)

'''Data preparing for batch update'''
df_contexts = pd.read_csv(csv_path)

if df_contexts.empty:
    print(f"No data found for batch update.\n")
else:
    df_contexts['timestamp'] = pd.to_datetime(df_contexts['id'].str.extract(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})')[0])
    df_contexts['delta_day'] = (today - df_contexts['timestamp']).dt.days

    gamma = 0.9
    df_final = pd.DataFrame()

    for gi, gv in df_contexts.groupby('timestamp'):
        n_samples = int(len(gv) * gamma ** gv['delta_day'].iloc[0])
        df_samples = gv.sample(n=n_samples, random_state=42)
    
        df_final = pd.concat([df_final, df_samples], axis=0)

    df_final = df_final.drop(columns=['timestamp', 'delta_day'])
    df_final = df_final.reset_index(drop=True)

    '''Make Index Loader'''
    batch_size = 64
    num_epochs = 1

    indices_torch = torch.tensor(df_final.index).type(torch.int64)
    indices_dataset = TensorDataset(indices_torch)
    indices_dataloader = DataLoader(indices_dataset, batch_size=batch_size, shuffle=True)
    print(f"Size: {len(indices_dataloader)}")

    feature_dict = baronow_model.model_dict['feature_columns']

    '''Batch Update'''
    for epoch in range(num_epochs):
        print(f"##### Epoch {epoch+1}/{num_epochs} #####")
        batch_update_model.run_batch_update(algorithm, indices_dataloader, {'base_x':feature_dict['base']}, pred_label='optimalAPI_path_time')
        batch_update_model.run_batch_update(algorithm, indices_dataloader, {'base_x':feature_dict['base'], 'timemachine_x': feature_dict['timemachine']}, pred_label='optimal_delta')
        batch_update_model.run_batch_update(algorithm, indices_dataloader, {'base_x':feature_dict['base'], 'timemachine_x': feature_dict['timemachine'], 'api_x':feature_dict['api']}, pred_label='optimal_delta')
        
    batch_update_model.save_weights()
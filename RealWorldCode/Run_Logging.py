import logging
import logging.handlers
import os
import datetime
import threading
import time
import json
import pandas as pd
import pytz
import re
import sys

file_name = os.path.abspath(__file__)
file_path = os.path.dirname(file_name)
base_path = '/'.join(file_path.split('/')[:[i for i, d in enumerate(file_path.split('/')) if 'BaroNowProject' in d][0]+1])
print(f"(file) {file_name.split('/')[-1]}, (base_path) {base_path}")

logging_path = os.path.join(base_path, 'logging')
sys.path.append(logging_path)
############################################################################################################
KST = pytz.timezone('Asia/Seoul')

'''log file to dataframe'''
def readAllLog():
    all_logs= []
    log_files = [file for file in os.listdir(logging_path) if file.endswith(".log")]
    
    for log_file in log_files:
        try:
            algorithm, date_with_ext = log_file.rsplit('_', 1)
            date = date_with_ext.replace(".log", "")
            print(f"Processing file: {log_file} | Model: {algorithm}, Date: {date}")
            
            df = readLog(algorithm, date)
            all_logs.append(df)
            
        except ValueError:
            print(f"Skipping file with unexpected format: {log_file}")
        
        if all_logs:
            combined_df = pd.concat(all_logs, ignore_index=True)
            return combined_df
        else:
            print("No valid logs found.")
            return pd.DataFrame()
            
def readLog(model_name, date):    
    logFile_path = os.path.join(logging_path, f"{model_name}_{date}.log")
    logs = []
    
    with open(logFile_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"\[(.*?)\] (.*)", line)
            if match:
                timestamp = match.group(1) # timestamp part
                log_data = match.group(2) # message part
                
                try:
                    log_json = json.loads(log_data)  # JSON-like part into dict
                    log_json['timestamp'] = timestamp
                    logs.append(log_json)
                    
                except json.JSONDecodeError:
                    print(f"Skipping line due to JSON decode error: {line}")
    
    if logs:
        df = pd.DataFrame(logs)
        return df
    else:
        print(f"No valid logs found in {logFile_path}")
        return pd.DataFrame()

'''filter the normal episodes'''
def episodeFilter(df): # status == 1이 정상종료
    # Dynamic Case에 의해 종료된 Episode 제외 Filtering
    new_episode_series = df.groupby('USER').apply(lambda x: x['Response_Env_History'].apply(lambda x: int(x['status'])==2).sum())
    id_no_new_episodes = new_episode_series[new_episode_series == 0].index.to_list()
    df_no_new_episodes = df[ df['USER'].isin(id_no_new_episodes) ]
    # df_no_new_episodes['Response_Env_History'].apply(lambda x: int(x['status'])==2).sum()

    # Leaving to Destination Case에 의해 정상 종료된 Episode만 Filtering
    df_no_new_episodes_last = df_no_new_episodes.groupby('USER').tail(1)
    df_leaving = df_no_new_episodes_last[df['Response_Env_History'].apply(lambda x: int(x['status'])==1)]
    df_normal_end = df_no_new_episodes[df_no_new_episodes['USER'].isin(df_leaving['USER'].to_list())]

    # 2줄이상 있는 episode만 filtering
    df_normal_end_group = df_normal_end.groupby('USER').size()
    df_over_two_line_idx = df_normal_end_group[df_normal_end_group >= 2].index.to_list()
    df_over_two_line = df_normal_end[df_normal_end['USER'].isin(df_over_two_line_idx)]
    
    return df_over_two_line

'''read last two contexts of each user'''
def lastTwo(df):
    if 'USER' not in df.columns:
        raise ValueError("DataFrame must contain a 'USER' column")

    # df = df.groupby('USER').filter(lambda x: len(x) >= 2)
    last2contexts = df.groupby('USER').tail(2)
    return last2contexts.reset_index(drop=True)

def lastTwoContexts(last_two_lines):
    df_contexts = last_two_lines['Response_Env_History'].apply(lambda x: x['contexts'])
    return df_contexts.apply(lambda x:pd.Series(x[0]))

class KSTFormatter(logging.Formatter):
    def converter(self, timestamp):
        utc_dt = datetime.datetime.fromtimestamp(timestamp, pytz.utc)
        kst_dt = utc_dt.astimezone(KST)
        return kst_dt.timetuple()
    
class logSave():    
    def __init__(self, dir, logname) -> None:
        self.model_name = logname
        self.dir = os.path.join(base_path, dir) # 코드가 있는 경로 하위에 로그 디렉토리 생성
        self.InitLogger()
        
    def InitLogger(self):
        formatter = KSTFormatter("[%(asctime)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        
        now = datetime.datetime.now(KST).strftime("%Y%m%d")
        log_File = os.path.join(self.dir, f"{self.model_name}_{now}.log")
        timedfilehandler = logging.handlers.TimedRotatingFileHandler(
            filename=log_File, when='midnight', interval=1, encoding='utf-8', backupCount=10
        )
        timedfilehandler.setFormatter(formatter)
        # timedfilehandler.suffix = "%Y%m%d"
        
        self.logger = logging.getLogger(self.model_name)
        
        if not self.logger.handlers:
            self.logger.addHandler(timedfilehandler)
            self.logger.setLevel(logging.INFO)
        
        self.delete_old_files(self.dir, 10)
        
        now = datetime.datetime.now(KST)
        self.toDay = "%04d-%02d-%02d" % (now.year, now.month, now.day)
        self.th_auto_delete = threading.Thread(target=self.on_auto_delete, daemon=True)
        self.th_auto_delete.start()

    def LogTextOut(self, log_dict):
        user_id = log_dict.get("USER")
        today = datetime.datetime.now(KST).strftime("%Y%m%d")
        log_file_name = f"{self.model_name}_{today}.log"
        log_file_path = os.path.join(self.dir, log_file_name)
        
        yesterday = (datetime.datetime.now(KST) - datetime.timedelta(days=1)).strftime("%Y%m%d")
        prev_log_file_name = f"{self.model_name}_{yesterday}.log"
        prev_log_file_path = os.path.join(self.dir, prev_log_file_name)
        
        user_exists_in_prev = False
        
        if os.path.exists(prev_log_file_path):
            with open(prev_log_file_path, "r", encoding="utf-8") as prev_log_file:
                for line in prev_log_file:
                    match = re.match(r"\[(.*?)\] (.*)", line)
                    if match:
                        log_data = json.loads(match.group(2))
                        if log_data.get("USER") == user_id:
                            user_exists_in_prev = True
                            break
        
        log_message = json.dumps(log_dict, ensure_ascii=False)
        
        if user_exists_in_prev:
            with open(prev_log_file_path, "a", encoding="utf-8") as prev_log_file:
                prev_log_file.write(f"{log_message}\n")
        else:
            self.logger.info(log_message)
                    
    def delete_old_files(self, path_target, days_elapsed):
        for f in os.listdir(path_target):
            f = os.path.join(path_target, f)
            if os.path.isfile(f):
                timestamp_now = datetime.datetime.now().timestamp()
                is_old = os.stat(f).st_mtime < timestamp_now - (days_elapsed * 24 * 60 * 60)
                if is_old:
                    try:
                        os.remove(f)
                        print(f, 'is deleted')
                    except OSError:
                        print(f, 'cannot delete')
                        
    def on_auto_delete(self): # 매일 자정에 자등으로 10일 이상 경과한 로그 파일을 삭제
        while True:
            now = datetime.datetime.now(KST)
            day = "%04d-%02d-%02d" % (now.year, now.month, now.day)
            if self.toDay != day:
                self.toDay = day
                self.delete_old_files(self.dir, 10)
            time.sleep(3600)
    
    def deleteAllLogs(self):
        confirm = input("Are you sure you want to delete all log files? Type 'yes' to confirm, 'no' to cancel: ").strip().lower()
        if confirm == 'yes':
            for f in os.listdir(self.dir):
                f_path = os.path.join(self.dir, f)
                if os.path.isfile(f_path):
                    try:
                        os.remove(f_path)
                        print(f"{f} has been deleted.")
                    except OSError as e:
                        print(f"Cannot delete {f}: {e}")
        else:
            print("Log deletion has been cancelled.")
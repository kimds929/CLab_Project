############################################################################################################
# (Path Setting)
import os
import sys
file_path = os.path.dirname(os.path.abspath(__file__))
base_path = '/'.join(file_path.split('/')[:[i for i, d in enumerate(file_path.split('/')) if 'BaroNowProject' in d][0]+1])
sys.path.append( base_path )
os.chdir(base_path)
print(f"base_path : {base_path}")

cold_start_path = os.path.join(base_path, 'cold_start');  sys.path.append( cold_start_path )
dataset_path = os.path.join(base_path, 'dataset');  sys.path.append( dataset_path )
env_path = os.path.join(base_path, 'environment');  sys.path.append( env_path )
logging_path = os.path.join(base_path, 'logging');  sys.path.append( logging_path )
module_path = os.path.join(base_path, 'module');    sys.path.append( module_path )
model_path = os.path.join(base_path, 'model');  sys.path.append( model_path )
weight_path = os.path.join(base_path, 'weight');    sys.path.append( weight_path )
############################################################################################################

##########################################################################################################################################################
"""model_py_file <- Put your model's name here"""
test_metod = 'offline'

from model.Algorithm3_Model import BaroNow_Algorithm3_Model as BaroNowModel
model_name = BaroNowModel.__name__
##########################################################################################################################################################
import requests
import json
import random
import urllib
import datetime
import holidays
import threading
import subprocess
import numpy as np
import pytz

from functools import wraps
import pandas as pd
import os
from datetime import timedelta

import time
import math

from model.test_model import EpisodeModel


import warnings
warnings.filterwarnings("ignore")
file_lock = threading.Lock()

# sungwoo key [Tmap, kakao]

tz = pytz.timezone('Asia/Seoul')

# 서울 중심로부터 66km 반경 내 편의점 중 랜덤하게 선택하여 좌표 가져옴
list_coordinates = [
    (37.5665, 126.9784),  # 서울 중심부(광화문)
    (37.8234, 127.1663),  # 1시 방향으로 33km
    (37.5659, 127.3528),  # 3시 방향
    (37.3093, 127.1650),  # 5시 방향
    (37.3093, 126.7918),  # 7시 방향
    (37.5659, 126.6040),  # 9시 방향
    (37.8234, 126.7905)   # 11시 방향
]

# random으로 좌표 선택 후 편의점 선택해주는 함수
def get_poi_names(tmap_appKey):

    while True:

        try:

            page = random.randint(1, 200)
            lat, lon = random.choice(list_coordinates)

            url = (
                "https://apis.openapi.sk.com/tmap/pois/search/around?version=1&centerLon="+str(lon)+"&centerLat="+str(lat)+
                "&categories=%ED%8E%B8%EC%9D%98%EC%A0%90&page=" + str(page) +
                "&count=200&radius=33&reqCoordType=WGS84GEO&resCoordType=WGS84GEO&multiPoint=N"
            )

            headers = {
                "accept": "application/json",
                "appKey": tmap_appKey
            }

            response = requests.get(url, headers=headers)
            # 상태 코드 확인
            if response.status_code != 200:
                continue  # 다음 반복으로 넘어감
            
            response_dict = response.json()
            
            # 'searchPoiInfo' 키 존재 여부 확인
            if 'searchPoiInfo' not in response_dict:
                print(f"'searchPoiInfo' 키가 응답에 없습니다. 응답 내용: {response_dict}")
                continue  # 다음 반복으로 넘어감
            
            # 'pois' 키 존재 여부 확인
            if 'pois' not in response_dict['searchPoiInfo']:
                print(f"'pois' 키가 응답에 없습니다. 응답 내용: {response_dict}")
                continue  # 다음 반복으로 넘어감
            
            # 'poi' 리스트가 비어 있는지 확인
            pois = response_dict['searchPoiInfo']['pois']['poi'] #'searchPoiInfo'->'pois'->'poi'->'name' 구조임
            if not pois:
                print("검색 결과가 없습니다.")
                continue  # 다음 반복으로 넘어감
            
            names = [poi["name"] for poi in pois]
            name = random.choice(names) # 편의점 랜덤 choice
            
            return name # return하면 while문 탈출함
        
        except Exception as e:
            print(f"예외 발생: {e}")
            continue


# id 생성해주는 함수
def generate_id(target_point, target_time):

    target_name = target_point['name'].replace(' ', '_')

    raw_id = f"{target_name}_{target_time}"

    return raw_id


# url에 필요한 것 생성 함수
def str_to_utf(facility_name=None): # for func:get_lon_lat

    encoded_keyword = urllib.parse.quote(facility_name, encoding='utf-8')

    return f"searchKeyword={encoded_keyword}"

# searchPoiInfo -> pois .. 장소에 관한 자세한 정보가 다 나옴
def get_location_from_place(facility_name, appKey): # for func:get_lon_lat

    url = "https://apis.openapi.sk.com/tmap/pois?version=1&"+str_to_utf(facility_name) +"&searchType=all&searchtypCd=A&reqCoordType=WGS84GEO&resCoordType=WGS84GEO&page=1&count=20&multiPoint=N&poiGroupYn=N"

    headers = {

        "Accept": "application/json",

        "appKey": appKey

    }


    response = requests.get(url, headers=headers)

    response_dict = json.loads(response.text)

    return response_dict

# get_location_from_place('CU 군포첨단산업단지점', appKeys[0][0])['searchPoiInfo']['pois']['poi'][0]


# 해당 이름 검색 결과 중 첫번째(가장 관련도 높은 주소)의 위도 경도 가져옴
def get_lon_lat(p_dict, tmap_appKey): 
    '''
    p_dict: 'CU 군포첨단산업단지점'
    tmap_appKey: tmap key
    '''
    facility_name = p_dict["name"]

    parsed_data = get_location_from_place(facility_name,tmap_appKey)

    pois = parsed_data["searchPoiInfo"]["pois"]["poi"]
    if pois:

        first_poi = pois[0]

        p_dict["lat"] = first_poi["frontLat"] # 위도

        p_dict["lon"] = first_poi["frontLon"] # 경도

# 경로와 각 구간의 시간 정보
def query_to_path_time(data): 
    
    lst = [
        (tuple(reversed(feature["geometry"]["coordinates"][0])),
         tuple(reversed(feature["geometry"]["coordinates"][-1])),
         feature["properties"]["time"])
        for feature in data["features"]
        if "description" in feature["properties"] and "time" in feature["properties"]
    ]
    return lst


# 경로만
def query_to_path(data): 

    valid_features = [
        feature
        for feature in data["features"]
        if "description" in feature["properties"] and "time" in feature["properties"]
    ]

    lst = [
        tuple(reversed(feature["geometry"]["coordinates"][0]))
        for feature in valid_features
    ]

    if valid_features: 
        last_feature = valid_features[-1]
        lst.append(tuple(reversed(last_feature["geometry"]["coordinates"][-1])))
    return lst


# 총 소요시간
def query_to_duration(query):

    return query['features'][0]['properties']['totalTime']


# https://skopenapi.readme.io/reference/%ED%83%80%EC%9E%84%EB%A8%B8%EC%8B%A0-%EC%9E%90%EB%8F%99%EC%B0%A8-%EA%B8%B8-%EC%95%88%EB%82%B4

# Tmap timemachine 정보
def query_timemachine(start_point, target_point, target_time, appKey):

    # url for timemachine

    url = "https://apis.openapi.sk.com/tmap/routes/prediction?version=1&resCoordType=WGS84GEO&reqCoordType=WGS84GEO&sort=index&callback=function"


    payload = {
        "routesInfo": {
            "departure": {
                "name": str_to_utf(start_point["name"]),
                "lon": start_point["lon"],
                "lat": start_point["lat"]
            },
            "destination": {
                "name": str_to_utf(target_point["name"]),
                "lon": target_point["lon"],
                "lat": target_point["lat"]
            },
            "predictionType": "departure",
            "predictionTime": target_time,
            "searchOption": "00",
            "tollgateCarType": "car",
            "trafficInfo": "N"
        }
    }
    
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "appKey": appKey
    }
    response = requests.post(url, json=payload, headers=headers)
    response_dict = json.loads(response.text)
    
    return response_dict

# kakao 경로 걸리는 시간
def query_to_path_time_kakao(data):
    routes = data['routes'][0]['sections']  # 첫 번째 경로의 섹션을 가져옴
    route_list = []
    for section in routes:
        # 각 섹션에서 도로 정보를 가져옴
        for road in section['roads']:
            # 시작 좌표와 끝 좌표, 소요 시간
            start_coordinate = (road['vertexes'][1], road['vertexes'][0])  # 위도, 경도
            end_coordinate = (road['vertexes'][-1], road['vertexes'][-2])  # 위도, 경도
            duration = road['duration']  # 소요 시간 (초)
            route_list.append((start_coordinate, end_coordinate, duration))
    return route_list

# kakao 경로
def query_to_path_kakao(data):
    routes = data['routes'][0]['sections']  # 첫 번째 경로의 섹션을 가져옴
    route_list = []
    if routes:  # 섹션이 존재하는 경우
        for section in routes:
            # 각 섹션에서 도로 정보를 가져옴
            for road in section['roads']:
                # 시작 좌표와 끝 좌표, 소요 시간
                start_coordinate = (road['vertexes'][1], road['vertexes'][0])  # 위도, 경도
                route_list.append(start_coordinate)
        # 마지막 section의 마지막 좌표 추가
        last_section = routes[-1]
        if last_section['roads']:  # 도로 정보가 있는 경우
            last_road = last_section['roads'][-1]
            # 마지막 도로의 마지막 좌표
            last_coordinate = (last_road['vertexes'][-1], last_road['vertexes'][-2])  # 위도, 경도
            route_list.append(last_coordinate)  # "Last Coordinate"라는 문자열과 함께 추가
    return route_list

# kakao 전체 경로
def query_to_duration_kakao(query):
    # 'routes' 키와 그 안의 첫 번째 요소 및 'summary' 키가 있는지 확인
    if 'routes' in query and len(query['routes']) > 0 and 'summary' in query['routes'][0]:
        return query['routes'][0]['summary']['duration']  # 첫 번째 경로의 요약 정보에서 총 소요 시간 (초) 가져옴
    else:
        # 문제가 있는 경우 query 출력
        print("경로 정보가 없거나 요약 정보가 없습니다. query 내용:")
        print(query)
        return None  # 오류가 발생했음을 나타내기 위해 None 반환

# 현재 api call 정보 가져오기
def query_realtime_kakao(start_point, target_point, API_KEY):

    # Kakao 길찾기 API URL

    url = "https://apis-navi.kakaomobility.com/v1/directions"
    

    # API 요청 헤더 설정

    headers = {

        "Authorization": f"KakaoAK {API_KEY}",
        "Content-Type": "application/json"

    }
    

    # 요청 파라미터 설정

    payload = {

        "origin": "{},{}".format(start_point["lon"], start_point["lat"]),

        "destination":  "{},{}".format(target_point["lon"], target_point["lat"]),

        "waypoints": [],

        "priority": "RECOMMEND",

        "car_fuel": "GASOLINE",

        "car_hipass": False

    }
    

    # API 요청

    response = requests.get(url, headers=headers, params=payload)
    

    # 결과 처리

    if response.status_code == 200:

        response_dict = response.json()
        return response_dict
    else:

        print("Error:", response.status_code, response.text)

        return None


# 요일:0~6 or 휴일:7 가져오기
def get_day_type(country='KR'):

    # 오늘 날짜 가져오기

    today = datetime.datetime.now().date()
    

    # 한국 공휴일 설정 (기본값은 'KR'으로 설정)

    country_holidays = holidays.CountryHoliday(country)


    # 공휴일 확인

    if today in country_holidays:

        return 7  # 공휴일이면 7 반환
    else:

        return today.weekday() # 월~일 순서로 0~6

# context -> model
def post_context_to_model(context):
    with open(f"{env_path}/online_data_input_test.json", 'w', encoding='utf-8') as file:
        json.dump(context.to_dict('records'), file, indent=4, ensure_ascii=False) 


def convert_to_second_format(row):
    def convert_point(point):
        return f"({point[0]}, {point[1]}, '{point[2]}')"
    
    def convert_path(path_list):
        return "[" + ", ".join([f"({p[0]}, {p[1]})" for p in path_list]) + "]"
    
    def convert_sub_path(sub_path_list):
        return "[" + ", ".join([f"(({sp[0][0]}, {sp[0][1]}), ({sp[1][0]}, {sp[1][1]}), {sp[2]})" for sp in sub_path_list]) + "]"
    
    # start_point, target_point, cur_point 등을 문자열로 변환
    row['start_point'] = convert_point(row['start_point'])
    row['target_point'] = convert_point(row['target_point'])
    row['cur_point'] = convert_point(row['cur_point'])
    
    # path_TimeMachine 및 sub_path_time_TimeMachine 등을 문자열로 변환
    row['path_TimeMachine'] = convert_path(row['path_TimeMachine'])
    row['sub_path_time_TimeMachine'] = convert_sub_path(row['sub_path_time_TimeMachine'])
    
    # sub_path_time_LastAPI를 문자열로 변환
    row['sub_path_time_LastAPI'] = convert_sub_path(row['sub_path_time_LastAPI'])
    
    return row


# 전달한 context를 바탕으로 model 실행
def run_predict_model():
    
    model_path = f'{base_path}/model'
    # with open(f'{model_path}/OnlinePrediction.py') as file:
    #     exec(file.read())

    subprocess.run(["python", f"{model_path}/{model_py_file}.py"])
    env_path = f"{base_path}/environment"
    response_from_model = None
    while(True):
        if 'online_data_output.json' in os.listdir(env_path):
            with open(f"{env_path}/online_data_output.json", "r") as file:
                response_from_model = json.load(file)
            break
    return response_from_model

# Reset Env
def reset_env():
    env_path = f"{base_path}/environment"
    os.listdir(env_path)
    if os.path.exists(f"{env_path}/online_data_input_test.json"):
        os.remove(f"{env_path}/online_data_input_test.json")
    if os.path.exists(f"{env_path}/online_data_output.json"):
        os.remove(f"{env_path}/online_data_output.json")

# context 껍질
def ouput_context(context):
    return {'user_id': None,
            'contexts': context,
            'isEnd': 0,
            'descriptions': None}

# moving agent
def move_randomly(lat, lon, speed, interval):
    # 랜덤한 방위각(0에서 2π 사이)
    bearing = random.uniform(0, 2 * math.pi)

    # 이동 거리 = 속도 * 시간 간격
    distance = speed * interval

    # 위도와 경도를 라디안으로 변환
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    # 지구의 반지름 (미터 단위)
    R = 6378137

    # 각 거리 = 실제 거리 / 지구 반지름
    angular_distance = distance / R

    # 새로운 위도 계산
    lat2 = math.asin(math.sin(lat1) * math.cos(angular_distance) +
                     math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing))

    # 새로운 경도 계산
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
                             math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2))

    # 라디안에서 도(degree)로 변환
    new_lat = math.degrees(lat2)
    new_lon = math.degrees(lon2)

    return new_lat, new_lon

# 시작 좌표 (예: 서울 시청)
start_lat = 37.5665
start_lon = 126.9780

current_lat = start_lat
current_lon = start_lon

# 에이전트의 속도 (예: 1.4 m/s, 보통 걷는 속도)
speed = 1.4  # meters per second

# 시뮬레이션 시간 (초 단위)
simulation_time = 60  # 1분간 시뮬레이션

# 위치 업데이트 간의 시간 간격 (초 단위)
interval = 1  # 1초마다 위치 업데이트

steps = int(simulation_time / interval)

for i in range(steps):
    current_lat, current_lon = move_randomly(current_lat, current_lon, speed, interval)
    print(f"Time {i * interval}s: 위도 {current_lat}, 경도 {current_lon}")
    time.sleep(interval)


# 시간과 호출 횟수를 관리하는 클래스
class TimeCheckerAuto:
    def __init__(self, appKey):
        self.tmap_appKey = appKey[0]
        self.kakao_appKey = appKey[1]
        
        self.initialize()  # 초기화 메서드 호출

    def initialize(self):
        
        self.min_diff = timedelta.max
        self.path_time_ans = 0
        # time.sleep(2*60-1)
        self.round = 0 # 초기 설정
        self.leaving_time = None
        self.context_history = pd.DataFrame()
        self.start_point = {"name": get_poi_names(self.tmap_appKey)} #random start point 설정
        # self.start_point = {"name": '포시즌마트'} #random start point 설정
        
        get_lon_lat(self.start_point, self.tmap_appKey)
        
        self.current_point = self.start_point
        
        self.target_point = {"name": get_poi_names(self.tmap_appKey)}
        # self.target_point = {"name": 'CU 트라팰리스점'}
        
        get_lon_lat(self.target_point, self.tmap_appKey)

        # 타겟 타임 설정을 위한 실시간 쿼리
        # query_rt = self.query_realtime(self.current_point, self.target_point, self.kakao_appKey)
        
        # # 실제 걸리는 시간(초)
        # duration = query_to_duration_kakao(query_rt)
        
        
        # 데이터 초기화
  
        
        
        # duration = query_to_duration(query_tm)
        duration = random.randint(1000,2500)
        kakao_cur_time = datetime.datetime.now(tz) # 현재시간 저장
        self.target_time = max(kakao_cur_time + timedelta(seconds=duration * 1.5), kakao_cur_time + timedelta(seconds=16 * 60))
        query_tm = self.query_timemachine(
            self.current_point,
            self.target_point,
            self.target_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            self.tmap_appKey,
        )
        self.id_hashed = generate_id(self.target_point, self.target_time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        self.start_time = kakao_cur_time 
        self.duration_tm = query_to_duration(query_tm)
        self.path_TimeMachine = query_to_path(query_tm)
        self.path_LastAPI = self.path_TimeMachine
        self.sub_path_time_TimeMachine = query_to_path_time(query_tm)
        self.call_time_TimeMachine = self.target_time - timedelta(seconds=self.duration_tm)
        self.req_leaving_time = self.target_time - timedelta(seconds=self.duration_tm + 8 * 60)
        # time_diff = abs(self.req_leaving_time - kakao_cur_time)
        # if time_diff < self.min_diff:
        #     self.min_diff = time_diff
        #     self.path_time_ans = query_to_duration_kakao(query_rt)

        self.df = pd.DataFrame(
            columns=[
                "id",
                "group",
                "transportation",
                "round",
                "start_time",
                "target_time",
                "start_point",
                "target_point",
                "call_time_TimeMachine",
                "path_TimeMachine",
                "sub_path_time_TimeMachine",
                "path_time_TimeMachine",
                "cur_time",
                "cur_point",
                "call_time_LastAPI",
                "call_point_LastAPI",
                "path_LastAPI",
                "path_time_LastAPI",
                "req_leaving_time",
                "sub_path_time_LastAPI",
                "weather",
                "path_time_ans",
                "event",
            ]
        )

        # 초기 데이터 추가
        initial_row = pd.DataFrame(
            [
                {
                    "id": self.id_hashed,
                    "group": "realtime",
                    "transportation": "car",
                    "round": self.round,
                    "start_time": self.start_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "target_time": self.target_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "start_point": (float(self.start_point["lat"]), float(self.start_point["lon"]), self.start_point["name"]),
                    "target_point": (float(self.target_point["lat"]), float(self.target_point["lon"]), self.target_point["name"]),
                    "call_time_TimeMachine": self.call_time_TimeMachine.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "path_TimeMachine": self.path_TimeMachine,
                    "sub_path_time_TimeMachine": self.sub_path_time_TimeMachine,
                    "path_time_TimeMachine": self.duration_tm,
                    "cur_time": self.start_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "cur_point": (float(self.current_point["lat"]), float(self.current_point["lon"]), self.current_point["name"]),
                    "call_time_LastAPI": np.nan,
                    "call_point_LastAPI": np.nan,
                    "path_LastAPI": np.nan,
                    "path_time_LastAPI": np.nan,
                    "req_leaving_time": self.req_leaving_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "sub_path_time_LastAPI":self.sub_path_time_TimeMachine,
                    "weather": np.nan,
                    "path_time_ans": 0,
                    "event": np.nan,
                }
            ]
        )

        self.df = pd.concat([self.df, initial_row], ignore_index=True)

    # kakao api call 정보가져오기
    def query_realtime(self, current_point, target_point, app_key):
        
        q = query_realtime_kakao(current_point, target_point, app_key)

        return q

    def query_timemachine(self, current_point, target_point, target_time_str, app_key):
        return query_timemachine(current_point, target_point, target_time_str, app_key)

    def new_context(self, query_rt):
        
        new_row = pd.DataFrame(
                [
                    {
                        "id": self.id_hashed,
                        "group": "realtime",
                        "transportation": "car",
                        "round": self.round,
                        "start_time": self.start_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "target_time": self.target_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "start_point": (float(self.start_point["lat"]), float(self.start_point["lon"]), self.start_point["name"]),
                        "target_point": (float(self.target_point["lat"]), float(self.target_point["lon"]), self.target_point["name"]),
                        "call_time_TimeMachine": self.call_time_TimeMachine.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "path_TimeMachine": self.path_TimeMachine,
                        "sub_path_time_TimeMachine": self.sub_path_time_TimeMachine,
                        "path_time_TimeMachine": self.duration_tm,
                        "cur_time": datetime.datetime.now(self.target_time.tzinfo).strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "cur_point": (float(self.current_point["lat"]), float(self.current_point["lon"]), self.current_point["name"]),
                        "call_time_LastAPI": datetime.datetime.now(self.target_time.tzinfo).strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "call_point_LastAPI": (float(self.current_point["lat"]), float(self.current_point["lon"]), self.current_point["name"]),
                        "path_LastAPI": query_to_path_kakao(query_rt),
                        "path_time_LastAPI": query_to_duration_kakao(query_rt),
                        "req_leaving_time": np.nan, #self.leaving_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "sub_path_time_LastAPI": query_to_path_time_kakao(query_rt),
                        "weather": np.nan,
                        "path_time_ans": 0,
                        "event": np.nan,
                    }
                ]
            )
        return new_row
    
    def start_checking(self):
        reset_env()

        response_history = []
        # output init
        api_call_time = None
        n_of_api_call = 0
        
        new_episodes = 0
        
        # 0 round
        context = self.df.apply(convert_to_second_format, axis=1)
        context_history = pd.concat([self.context_history, context], axis=0)
        #####################################
        post_context_to_model(context)      # post context to model
        response = run_predict_model()
        #####################################
        
        self.round += 1
        
        while True:
            time.sleep(50)  # 2분 대기
            print(f"{self.round - 1} {datetime.datetime.strftime(datetime.datetime.now(self.target_time.tzinfo), '%Y-%m-%d %H:%M:%S')}", end ='\t')
            print(response)

            if response['api_call_time'] is not None:
                api_call_time = datetime.datetime.strptime(response['api_call_time'], "%Y-%m-%dT%H:%M:%S%z")
                
            if response['leaving_time'] is not None:
                self.leaving_time = datetime.datetime.strptime(response['leaving_time'], "%Y-%m-%dT%H:%M:%S%z")
            
            if response['new_episodes'] == 1:
                new_episodes = 1
                print('New_Episodes.')
                break
            # api call 하게되는 case
            if (n_of_api_call == 0) and (api_call_time is not None) and (datetime.datetime.now(self.target_time.tzinfo) > api_call_time):
                if self.round == 1:
                    print('next episode')
                    break
                print('api_call')
                query_rt = self.query_realtime(self.current_point, self.target_point, self.kakao_appKey)
                print('1')
                context = self.new_context(query_rt)
                print('2')
                n_of_api_call += 1  # api_call 횟수
                
                # api call한 시점에 허용 범위내 들어와있을 경우 바로 출발
                if datetime.datetime.strptime(context['cur_time'].item(), "%Y-%m-%dT%H:%M:%S%z") + timedelta(seconds=context['path_time_LastAPI'].item()) > datetime.datetime.strptime(context['target_time'].item(), "%Y-%m-%dT%H:%M:%S%z") - timedelta(seconds=13*60):
                    print('4-1')
                    context_history = pd.concat([context_history, context], axis=0)
                    print('Leaving to Destination (Not Predict delta_path_time). End_of_episodes.')
                    query_rt = self.query_realtime(self.current_point, self.target_point, self.kakao_appKey)
                    print('5-1')
                    context = self.new_context(query_rt)
                    print('6-1')
                    context = context.apply(convert_to_second_format, axis=1)
                    context_history = pd.concat([context_history, context], axis=0)
                    break
            # no api case
            else:
                print('no_api')
                context = self.df
                print('test1-1')
                context['group']= 'realtime'
                print('test1-2')
                context['cur_time']= datetime.datetime.now(self.target_time.tzinfo).strftime("%Y-%m-%dT%H:%M:%S%z")
                print('test1-3')
                context['round']= self.round
                print('test1-4')
                context['req_leaving_time']=self.req_leaving_time.strftime("%Y-%m-%dT%H:%M:%S%z")
                print('test1-5')
                if pd.isna(context['call_time_LastAPI'].iloc[0]):
                    print('test2-1')
                    context['call_time_LastAPI']= context_history.iloc[0]['cur_time']
                    print('test2-2')
                    context['call_point_LastAPI']= context_history.iloc[0]['cur_point']
                    print('test2-3')
                    context['path_LastAPI']= context_history.iloc[0]['path_TimeMachine']
                    print('test2-4')
                    context['path_time_LastAPI']= context_history.iloc[0]['path_time_TimeMachine']
                    print('test2-5')
                    
                else:
                    for api_col in ['call_time_LastAPI', 'call_point_LastAPI', 'path_LastAPI', 'path_time_LastAPI']:
                        print('test3-1')
                        print(context[api_col])
                        print(context_history.iloc[-1,:][api_col])  # no_api call의 경우 이전 api 정보를 최신 정보로 사용
                        # context[api_col] = context_history.iloc[-1,:][api_col]  # no_api call의 경우 이전 api 정보를 최신 정보로 사용
                        context.at[context.index[0], api_col] = context_history.iloc[-1,:][api_col]
                        print('test3-2')
                    
            # predict leaving time 도래시
            if (self.leaving_time is not None) and (datetime.datetime.now(self.target_time.tzinfo) >= self.leaving_time):
                
                context = self.df
                context['group']= 'realtime'
                context['cur_time']= datetime.datetime.now(self.target_time.tzinfo).strftime("%Y-%m-%dT%H:%M:%S%z")
                context['round']= self.round
                context['req_leaving_time']=self.req_leaving_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                
                if pd.isna(context['call_time_LastAPI'].iloc[0]):
                    context['call_time_LastAPI']= context_history.iloc[0]['cur_time']
                    context['call_point_LastAPI']= context_history.iloc[0]['cur_point']
                    context['path_LastAPI']= context_history.iloc[0]['path_TimeMachine']
                    context['path_time_LastAPI']= context_history.iloc[0]['path_time_TimeMachine']
                else:
                    for api_col in ['call_time_LastAPI', 'call_point_LastAPI', 'path_LastAPI', 'path_time_LastAPI']:
                        # context[api_col] = context_history.iloc[-1,:][api_col]  # no_api call의 경우 이전 api 정보를 최신 정보로 사용
                        print('test4-1')
                        print(context[api_col])
                        print(context_history.iloc[-1,:][api_col])  # no_api call의 경우 이전 api 정보를 최신 정보로 사용
                        # context[api_col] = context_history.iloc[-1,:][api_col]  # no_api call의 경우 이전 api 정보를 최신 정보로 사용
                        context.at[context.index[0], api_col] = context_history.iloc[-1,:][api_col]
                        print('test4-2')

                context_history = pd.concat([context_history, context], axis=0)
                print('Leaving to Destination. End_of_episodes.')
                query_rt = self.query_realtime(self.current_point, self.target_point, self.kakao_appKey)
                print('5-1')
                context = self.new_context(query_rt)
                print('5-2')
                context = context.apply(convert_to_second_format, axis=1)
                context_history = pd.concat([context_history, context], axis=0)
                break

            context=context.apply(convert_to_second_format, axis=1)
            context_history = pd.concat([context_history, context], axis=0,ignore_index=True)
            ##########################################
            post_context_to_model(context_history)
            response = run_predict_model()
            #########################################
            self.round += 1
            # records_history
            response_history.append(response)
            
            path_time=context["path_time_LastAPI"].iloc[0]
            if pd.isna(path_time):
                duration = int(self.duration_tm)
            else:
                duration = int(path_time)

            if self.target_time - timedelta(seconds=duration) <= datetime.datetime.now(self.target_time.tzinfo):  # 종료 조건 : 지금 나가면 약속시간 이후 도착 예정일 때
                query_rt = self.query_realtime(self.current_point, self.target_point, self.kakao_appKey)
                print('4-3')
                context = self.new_context(query_rt)
                print('5-3')
                context = context.apply(convert_to_second_format, axis=1)
                context_history = pd.concat([context_history, context], axis=0)
                break
        
        return context_history
    
    def save_to_csv(self, file_path):
        """Saves the checker's DataFrame to an Excel file. If the file exists, append to it. 

        If it doesn't exist, create a new file.
        """
        last_round_value = self.df['round'].iloc[-1]
        if last_round_value > 1:
            with file_lock:
                if os.path.exists(file_path):
                    existing_df = pd.read_csv(file_path, encoding="utf-8-sig", on_bad_lines="skip")
                    combined_df = pd.concat([existing_df, self.df], ignore_index=True)
                else:
                    combined_df = self.df

                combined_df.to_csv(file_path, index=False, encoding="utf-8-sig")



def run_time_checker(appkey, sleep_condition=False):
    while True:
        try:
            # TimeCheckerAuto 인스턴스 생성
            a = TimeCheckerAuto(appkey)
            # start_checking 메서드 실행
            context_history=a.start_checking()
            # 결과를 CSV 파일로 저장
            last_context = context_history.iloc[-1,:]
            if not pd.isna(last_context['path_time_LastAPI']):

                arrival_time = datetime.datetime.strptime(last_context['cur_time'], "%Y-%m-%dT%H:%M:%S%z") + timedelta(seconds=last_context['path_time_LastAPI'])
                residual_arrival_time = (arrival_time - datetime.datetime.strptime(last_context['target_time'], "%Y-%m-%dT%H:%M:%S%z") ).total_seconds()/60
                print(f"Is in allowance range? : {-13 < residual_arrival_time < 0}, {residual_arrival_time}")
                with open("/home/kimds929/BaroNowProject/result.txt", "a") as file:  # 'a' 모드는 파일에 덧붙여 쓰기
                    file.write(f"Is in allowance range? : {-13 < residual_arrival_time < 0}, {residual_arrival_time}\n")
            # a.save_to_csv(datetime.datetime.now(tz).strftime('%Y%m%d') + '.csv')
        except KeyError as e:
            print(f"KeyError 발생: {e}. 초기화 시도 중...")
            # 예외 발생 시 초기화 시도
            continue

        except Exception as e:
            print(f"알 수 없는 에러 발생: {e}. 초기화 시도 중...")
            # 다른 예외 발생 시 초기화 시도
            continue

        # sleep_condition이 True인 경우 추가 동작
        if sleep_condition:
            current_time = datetime.datetime.now(tz).time()

            # 현재 시간이 새벽 1시에서 5시 사이인지 확인
            if current_time >= datetime.time(1, 0) and current_time < datetime.time(5, 0):
                time.sleep(4 * 60 * 60)  # 4시간 동안 대기 (단위: 초)

    
    # model에 context를 전달


if __name__ == "__main__":
    
    appKeys =  [('jICW6XpKkB9hSdzIJrySU4EtcJ7JzDTZBBPDeHo7', 'bfeae872ac9a35817ff18366aeb852dd')] 
    run_time_checker(appKeys[0])
    
    # # 스레드를 사용하여 15개의 인스턴스를 병렬로 실행
    # threads = []

    # for key in appKeys:
    #     for i in range(15):
    #         # 첫 번째 10개의 스레드에는 sleep 조건을 설정
    #         if i < 10:
    #             thread = threading.Thread(target=run_time_checker, args=(key, True,))
    #         else:
    #             thread = threading.Thread(target=run_time_checker, args=(key, False,))

    #         threads.append(thread)
    #         thread.start()

    #         time.sleep(5)

    # # 모든 스레드가 완료될 때까지 대기
    # for thread in threads:
    #     thread.join()

    
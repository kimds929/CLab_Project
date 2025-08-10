############################################################################################################
# (Path Setting)
import os
import sys
file_name = os.path.abspath(__file__)
file_path = os.path.dirname(file_name)
base_path = '/'.join(file_path.replace('\\','/').split('/')[:[i for i, d in enumerate(file_path.replace('\\','/').split('/')) if 'BaroNowProject' in d][0]+1])
file_path.split('/')[:[i for i, d in enumerate(file_path.split('/')) if 'BaroNowProject' in d][0]+1]
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
import time

from datetime import datetime, timedelta

import json
import ast
from IPython.display import clear_output

# BaroNow Prediction Class Load
from Public_Run_Prediction import PublicPrediction
# from Run_RuleBased import BaroNowRuleBased


##########################################################################################################################################################
import requests
import json
import random
import urllib
import threading
import pytz

import math


file_lock = threading.Lock()


tz = pytz.timezone('Asia/Seoul')

# tmap: 반경안에 있는 카페 가져오기, 좌표가져오기,  타임머신 데이터, 실시간 교통상황 <- 이거만 추가
# odsay: 실시간 교통상황, 타임머신 기능 없음

# random으로 좌표 선택 후 편의점 선택해주는 함수
# def get_poi_names(tmap_appKey):
#     # 서울 중심로부터 66km 반경 내 편의점 중 랜덤하게 선택하여 좌표 가져옴
#     list_coordinates = [
#         (37.5665, 126.9784),  # 서울 중심부(광화문)
#         (37.8234, 127.1663),  # 1시 방향으로 33km
#         (37.5659, 127.3528),  # 3시 방향
#         (37.3093, 127.1650),  # 5시 방향
#         (37.3093, 126.7918),  # 7시 방향
#         (37.5659, 126.6040),  # 9시 방향
#         (37.8234, 126.7905)   # 11시 방향
#     ]
#     while True:

#         try:

#             page = random.randint(1, 200)
#             lat, lon = random.choice(list_coordinates)

#             url = (
#                 "https://apis.openapi.sk.com/tmap/pois/search/around?version=1&centerLon="+str(lon)+"&centerLat="+str(lat)+
#                 "&categories=%EC%B9%B4%ED%8E%98&page=" + str(page) +
#                 "&count=200&radius=33&reqCoordType=WGS84GEO&resCoordType=WGS84GEO&multiPoint=N"
#             )

#             headers = {
#                 "accept": "application/json",
#                 "appKey": tmap_appKey
#             }

#             response = requests.get(url, headers=headers)
#             # 상태 코드 확인
#             if response.status_code != 200:
#                 continue  # 다음 반복으로 넘어감
            
#             response_dict = response.json()
            
#             # 'searchPoiInfo' 키 존재 여부 확인
#             if 'searchPoiInfo' not in response_dict:
#                 print(f"'searchPoiInfo' 키가 응답에 없습니다. 응답 내용: {response_dict}")
#                 continue  # 다음 반복으로 넘어감
            
#             # 'pois' 키 존재 여부 확인
#             if 'pois' not in response_dict['searchPoiInfo']:
#                 print(f"'pois' 키가 응답에 없습니다. 응답 내용: {response_dict}")
#                 continue  # 다음 반복으로 넘어감
            
#             # 'poi' 리스트가 비어 있는지 확인
#             pois = response_dict['searchPoiInfo']['pois']['poi'] #'searchPoiInfo'->'pois'->'poi'->'name' 구조임
#             if not pois:
#                 print("검색 결과가 없습니다.")
#                 continue  # 다음 반복으로 넘어감
            
#             names = [(poi["name"], poi['frontLat'],poi['frontLon']) for poi in pois]
#             res = tuple(random.choice(names))
#             return res # return하면 while문 탈출함
        
#         except Exception as e:
#             print(f"예외 발생: {e}")
#             continue

seoul_districts = [
    {'name': 'Gangnam-gu', 'lat': 37.5172, 'lon': 127.0473, 'radius': 3.54, 'population_live': 561052, 'population_work': 866835},
    {'name': 'Gangdong-gu', 'lat': 37.5301, 'lon': 127.1238, 'radius': 2.80, 'population_live': 427418, 'population_work': 165739},
    {'name': 'Gangbuk-gu', 'lat': 37.6396, 'lon': 127.0250, 'radius': 2.74, 'population_live': 316085, 'population_work': 67912},
    {'name': 'Gangseo-gu', 'lat': 37.5509, 'lon': 126.8495, 'radius': 3.63, 'population_live': 558227, 'population_work': 161582},
    {'name': 'Gwanak-gu', 'lat': 37.4784, 'lon': 126.9516, 'radius': 3.07, 'population_live': 502089, 'population_work': 99031},
    {'name': 'Gwangjin-gu', 'lat': 37.5384, 'lon': 127.0826, 'radius': 2.33, 'population_live': 364671, 'population_work': 119482},
    {'name': 'Guro-gu', 'lat': 37.4955, 'lon': 126.8874, 'radius': 2.53, 'population_live': 418276, 'population_work': 240209},
    {'name': 'Geumcheon-gu', 'lat': 37.4569, 'lon': 126.8958, 'radius': 2.03, 'population_live': 240230, 'population_work': 123987},
    {'name': 'Nowon-gu', 'lat': 37.6543, 'lon': 127.0565, 'radius': 3.36, 'population_live': 535282, 'population_work': 89123},
    {'name': 'Dobong-gu', 'lat': 37.6659, 'lon': 127.0318, 'radius': 2.56, 'population_live': 332423, 'population_work': 55678},
    {'name': 'Dongdaemun-gu', 'lat': 37.5743, 'lon': 127.0398, 'radius': 2.13, 'population_live': 346770, 'population_work': 152345},
    {'name': 'Dongjak-gu', 'lat': 37.5124, 'lon': 126.9395, 'radius': 2.28, 'population_live': 397104, 'population_work': 105678},
    {'name': 'Mapo-gu', 'lat': 37.5638, 'lon': 126.9085, 'radius': 2.76, 'population_live': 381330, 'population_work': 250789},
    {'name': 'Seodaemun-gu', 'lat': 37.5793, 'lon': 126.9368, 'radius': 2.36, 'population_live': 309006, 'population_work': 140567},
    {'name': 'Seocho-gu', 'lat': 37.4836, 'lon': 127.0327, 'radius': 3.87, 'population_live': 427515, 'population_work': 650123},
    {'name': 'Seongdong-gu', 'lat': 37.5635, 'lon': 127.0361, 'radius': 2.32, 'population_live': 303965, 'population_work': 220456},
    {'name': 'Seongbuk-gu', 'lat': 37.5894, 'lon': 127.0168, 'radius': 2.80, 'population_live': 441618, 'population_work': 110345},
    {'name': 'Songpa-gu', 'lat': 37.5146, 'lon': 127.1056, 'radius': 3.28, 'population_live': 677489, 'population_work': 240789},
    {'name': 'Yangcheon-gu', 'lat': 37.5271, 'lon': 126.8560, 'radius': 2.36, 'population_live': 464148, 'population_work': 98234},
    {'name': 'Yeongdeungpo-gu', 'lat': 37.5260, 'lon': 126.8962, 'radius': 2.80, 'population_live': 368402, 'population_work': 350567},
    {'name': 'Yongsan-gu', 'lat': 37.5324, 'lon': 126.9901, 'radius': 2.64, 'population_live': 243243, 'population_work': 210456},
    {'name': 'Eunpyeong-gu', 'lat': 37.6177, 'lon': 126.9227, 'radius': 3.07, 'population_live': 481572, 'population_work': 88123},
    {'name': 'Jongno-gu', 'lat': 37.5729, 'lon': 126.9794, 'radius': 2.76, 'population_live': 151767, 'population_work': 290345},
    {'name': 'Jung-gu', 'lat': 37.5575, 'lon': 126.9941, 'radius': 1.78, 'population_live': 126217, 'population_work': 350678},
    {'name': 'Jungnang-gu', 'lat': 37.6065, 'lon': 127.0928, 'radius': 2.43, 'population_live': 397245, 'population_work': 78234},
]
gyeonggi_cities = [
    {'name': 'Suwon-si', 'lat': 37.2636, 'lon': 127.0286, 'radius': 6.20, 'population_live': 1194296, 'population_work': 600000},
    {'name': 'Seongnam-si', 'lat': 37.4200, 'lon': 127.1265, 'radius': 6.72, 'population_live': 944286, 'population_work': 450000},
    {'name': 'Goyang-si', 'lat': 37.6584, 'lon': 126.8320, 'radius': 9.22, 'population_live': 1073069, 'population_work': 350000},
    {'name': 'Yongin-si', 'lat': 37.2411, 'lon': 127.1776, 'radius': 13.72, 'population_live': 1069925, 'population_work': 400000},
    {'name': 'Bucheon-si', 'lat': 37.5037, 'lon': 126.7660, 'radius': 4.12, 'population_live': 848100, 'population_work': 200000},
    {'name': 'Ansan-si', 'lat': 37.3219, 'lon': 126.8309, 'radius': 6.90, 'population_live': 679185, 'population_work': 250000},
    {'name': 'Anyang-si', 'lat': 37.3943, 'lon': 126.9568, 'radius': 4.31, 'population_live': 573469, 'population_work': 180000},
    {'name': 'Namyangju-si', 'lat': 37.6360, 'lon': 127.2165, 'radius': 12.07, 'population_live': 713755, 'population_work': 100000},
    {'name': 'Hwaseong-si', 'lat': 37.1996, 'lon': 126.8310, 'radius': 14.80, 'population_live': 832077, 'population_work': 150000},
    {'name': 'Pyeongtaek-si', 'lat': 36.9920, 'lon': 127.1128, 'radius': 12.01, 'population_live': 493118, 'population_work': 120000},
    {'name': 'Uijeongbu-si', 'lat': 37.7380, 'lon': 127.0450, 'radius': 5.10, 'population_live': 450506, 'population_work': 80000},
    {'name': 'Paju-si', 'lat': 37.7602, 'lon': 126.7794, 'radius': 14.62, 'population_live': 453752, 'population_work': 70000},
    {'name': 'Siheung-si', 'lat': 37.3795, 'lon': 126.8031, 'radius': 6.55, 'population_live': 427610, 'population_work': 90000},
    {'name': 'Gunpo-si', 'lat': 37.3622, 'lon': 126.9350, 'radius': 3.40, 'population_live': 286485, 'population_work': 60000},
    {'name': 'Gwangmyeong-si', 'lat': 37.4772, 'lon': 126.8644, 'radius': 3.50, 'population_live': 346376, 'population_work': 50000},
    {'name': 'Osan-si', 'lat': 37.1499, 'lon': 127.0770, 'radius': 3.69, 'population_live': 223894, 'population_work': 40000},
    {'name': 'Icheon-si', 'lat': 37.2796, 'lon': 127.4429, 'radius': 12.10, 'population_live': 209339, 'population_work': 30000},
    {'name': 'Anseong-si', 'lat': 37.0079, 'lon': 127.2708, 'radius': 13.28, 'population_live': 184515, 'population_work': 20000},
    {'name': 'Gimpo-si', 'lat': 37.6157, 'lon': 126.7150, 'radius': 9.38, 'population_live': 469420, 'population_work': 60000},
    {'name': 'Gwangju-si', 'lat': 37.4138, 'lon': 127.2574, 'radius': 11.70, 'population_live': 365001, 'population_work': 30000},
    {'name': 'Yangju-si', 'lat': 37.7850, 'lon': 127.0450, 'radius': 9.92, 'population_live': 230090, 'population_work': 20000},
    {'name': 'Dongducheon-si', 'lat': 37.9158, 'lon': 127.0539, 'radius': 5.51, 'population_live': 94765, 'population_work': 10000},
    {'name': 'Uiwang-si', 'lat': 37.3446, 'lon': 126.9682, 'radius': 4.14, 'population_live': 158436, 'population_work': 20000},
    {'name': 'Hanam-si', 'lat': 37.5400, 'lon': 127.2056, 'radius': 5.44, 'population_live': 275143, 'population_work': 30000},
    {'name': 'Pocheon-si', 'lat': 37.8947, 'lon': 127.2001, 'radius': 16.21, 'population_live': 150000, 'population_work': 15000},
    {'name': 'Yeoju-si', 'lat': 37.2980, 'lon': 127.6375, 'radius': 13.91, 'population_live': 110902, 'population_work': 10000},
    {'name': 'Guri-si', 'lat': 37.5985, 'lon': 127.1398, 'radius': 3.26, 'population_live': 192292, 'population_work': 20000},
    {'name': 'Gwacheon-si', 'lat': 37.4292, 'lon': 126.9871, 'radius': 3.38, 'population_live': 70935, 'population_work': 15000},
    {'name': 'Yangpyeong-gun', 'lat': 37.4890, 'lon': 127.4914, 'radius': 16.72, 'population_live': 116197, 'population_work': 10000},
    {'name': 'Gapyeong-gun', 'lat': 37.8315, 'lon': 127.5101, 'radius': 16.37, 'population_live': 63033, 'population_work': 5000},
    {'name': 'Yeoncheon-gun', 'lat': 38.0960, 'lon': 127.0750, 'radius': 14.86, 'population_live': 44193, 'population_work': 4000},
]

def sample_region(option):
    # 옵션 검증
    if option not in ['live', 'work']:
        raise ValueError("옵션은 'live' 또는 'work' 중 하나여야 합니다.")
    if option == 'live':
        seoul_population = sum(district['population_live'] for district in seoul_districts)
        gyeonggi_population = sum(city['population_live'] for city in gyeonggi_cities)
    else:  # option == 'work'
        seoul_population = sum(district['population_work'] for district in seoul_districts)
        gyeonggi_population = sum(city['population_work'] for city in gyeonggi_cities)
    choice = random.choices(
        ['seoul', 'gyeonggi'],
        weights=[seoul_population, gyeonggi_population],
        k=1
    )[0]
    other = 'gyeonggi' if choice == 'seoul' else 'seoul'
    return (choice, other)

def get_poi_names(tmap_appKey, region='seoul', option='live'):
    # 데이터 선택
    if region == 'seoul':
        locations = seoul_districts
    elif region == 'gyeonggi':
        locations = gyeonggi_cities
    else:
        raise ValueError("Invalid region. Choose 'seoul' or 'gyeonggi'.")
    # 가중치 설정
    if option == 'live':
        weights = [loc['population_live'] for loc in locations]
    elif option == 'work':
        weights = [loc['population_work'] for loc in locations]
    else:
        raise ValueError("Invalid option. Choose 'live' or 'work'.")

    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]

    # 가중치 기반 랜덤 선택
    location = random.choices(locations, weights=probabilities, k=1)[0]
    lat = location['lat']
    lon = location['lon']
    radius = location['radius']
    radius = math.ceil(radius)
    count_per_page = 100  # 페이지당 POI 수
    # 첫 요청으로 페이지 정보 확인
    url = (
        "https://apis.openapi.sk.com/tmap/pois/search/around?version=1"
        f"&centerLon={lon}&centerLat={lat}"
        "&categories=%ED%8E%B8%EC%9D%98%EC%A0%90"
        "&page=1"
        f"&count={count_per_page}"
        f"&radius={radius}"
        "&reqCoordType=WGS84GEO"
        "&resCoordType=WGS84GEO"
        "&multiPoint=N"
    )
    headers = {
        "accept": "application/json",
        "appKey": tmap_appKey
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data: {response.status_code}")

    response_dict = response.json()
    if 'searchPoiInfo' not in response_dict:
        raise Exception(f"'searchPoiInfo' key not found in response: {response_dict}")
    
    search_info = response_dict['searchPoiInfo']
    total_count = int(search_info['totalCount'])  # 전체 POI 수
    
    total_pages = (total_count // count_per_page) + (1 if total_count % count_per_page > 0 else 0)  # 총 페이지 수

    if total_pages == 0:
        raise Exception("No data available for the given query.")

    # 랜덤 페이지 선택
    page = random.randint(1, total_pages)

    if page == total_pages:
        # 마지막 페이지일 경우 데이터 개수 계산
        last_page_count = total_count % count_per_page or count_per_page
    else:
        last_page_count = count_per_page

    # 요청 URL 수정
    url = url.replace("&page=1", f"&page={page}")
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch data for page {page}: {response.status_code}")
    
    response_dict = response.json()
    pois = response_dict['searchPoiInfo']['pois']['poi']
    if not pois:
        raise Exception("No POI data available even on a valid page.")

    # 랜덤 POI 선택
    if page == total_pages:
        pois = pois[:last_page_count]  # 마지막 페이지는 유효한 데이터만 사용
    names = [(poi["name"], poi['frontLat'], poi['frontLon']) for poi in pois]
    res = tuple(random.choice(names))
    return res


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



##############odsay#################
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

# kakao 현재 api call 정보 가져오기
def query_realtime_kakao(start_point, target_point, API_KEY):
    print('real_api_kakao')
    lat_start, lon_start , name_start = ast.literal_eval(start_point)
    lat_target, lon_target, name_target = ast.literal_eval(target_point)
    
    # Kakao 길찾기 API URL
    url = "https://apis-navi.kakaomobility.com/v1/directions"
    # API 요청 헤더 설정
    headers = {
        "Authorization": f"KakaoAK {API_KEY}",
        "Content-Type": "application/json"
    }
    # 요청 파라미터 설정
    payload = {
        "origin": "{},{}".format(lon_start, lat_start),
        "destination":  "{},{}".format(lon_target, lat_target),
        "waypoints": [],
        "priority": "RECOMMEND",
        "avoid" : "roadevent",
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

################Tmap###################
# tmap 경로 걸리는 시간
def query_to_path_time_tmap(data): # 경로와 각 구간의 시간 정보
    lst = [
        feature["properties"]["time"] for feature in data["features"]
        if "description" in feature["properties"] and "time" in feature["properties"]
    ]
    return lst

# tmap 경로
def query_to_path_tmap(data):  # 경로만
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

# tmap 전체 경로
def query_to_duration_tmap(query): # 총 소요시간
    return query['features'][0]['properties']['totalTime']

# tmap 현재 api call 정보 가져오기
def query_realtime_tmap(start_point, target_point, API_KEY):
    print('real_api_tmap')
    lat_start, lon_start , name_start = ast.literal_eval(start_point)
    lat_target, lon_target, name_target = ast.literal_eval(target_point)
    # TMAP 길찾기 API URL
    url = "https://apis.openapi.sk.com/tmap/routes?version=1&callback=function"

    # API 요청 헤더 설정
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "appKey": API_KEY
    }

    # 요청 파라미터 설정
    payload = {
        "endX": lon_target, #target_point["lon"],
        "endY": lat_target, #target_point["lat"],
        "startX": lon_start, #start_point["lon"],
        "startY": lat_start #start_point["lat"]
    }

    # API 요청 (TMAP은 POST 요청 사용)
    response = requests.post(url, headers=headers, json=payload)

    # 결과 처리
    if response.status_code == 200:
        response_dict = response.json()
        return response_dict
    else:
        print("Error:", response.status_code, response.text)
        return None

###############odsay###################

# odsay 전체 정보 가져오기
def query_realtime_odsay(start_point, target_point, API_KEY):
    """
    ODsay API를 이용한 대중교통 경로 탐색 (간소화)
    """
    # ODsay API URL
    
    url = "https://api.odsay.com/v1/api/searchPubTransPath"
   
    # 요청 파라미터 설정 (간소화)
    params = {
        "SX": start_point["lon"],  # 출발지 경도
        "SY": start_point["lat"],  # 출발지 위도
        "EX": target_point["lon"],  # 도착지 경도
        "EY": target_point["lat"],  # 도착지 위도
        "apiKey": API_KEY,
    }
    
    # API 요청
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        response_dict = response.json()
        return response_dict.get("result", {})  # 결과 반환 (없으면 빈 딕셔너리)
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return {}


# odsay 경로 총 소요시간
def query_to_duration_odsay(data):
    """
    총 소요 시간 추출 (분 단위)
    """
    if not data or "path" not in data or not data["path"]:
        print("경로 데이터가 없거나 잘못된 형식입니다.")
        return None

    first_path = data["path"][0]

    try:
        return first_path["info"]["totalTime"]  # 분 단위
    except KeyError:
        print("총 소요 시간 정보를 찾을 수 없습니다.")
        return None

# odsay 경로
def query_to_path_odsay(data):
    """
    도보 구간을 제외한 각 구간의 (시작 지점, 종료 지점) 쌍 추출
    """
    path_pairs = []
    if not data or "path" not in data or not data["path"]:
        print("경로 데이터가 없거나 잘못된 형식입니다.")
        return []

    first_path = data["path"][0]
    sub_paths = first_path.get("subPath", [])

    for sub_path in sub_paths:
        # 도보 구간 제외
        if sub_path.get("trafficType") != 3:
            startX = sub_path.get("startX")
            startY = sub_path.get("startY")
            endX = sub_path.get("endX")
            endY = sub_path.get("endY")
            if startX is not None and startY is not None and endX is not None and endY is not None:
                start_point = (startY, startX)  # (위도, 경도)
                end_point = (endY, endX)
                path_pairs.append((start_point, end_point))

    return path_pairs

# odsay subpath 시간
def query_to_sub_path_time_odsay(data):
    """
    도보 구간을 제외한 각 구간의 소요 시간 추출 (초 단위)
    """
    times = []
    if not data or "path" not in data or not data["path"]:
        print("경로 데이터가 없거나 잘못된 형식입니다.")
        return []

    first_path = data["path"][0]
    sub_paths = first_path.get("subPath", [])

    for sub_path in sub_paths:
        # 도보 구간 제외
        if sub_path.get("trafficType") != 3:
            section_time = sub_path.get("sectionTime")
            if section_time is not None:
                times.append(section_time) 

    return times

######################################
def convert_to_second_format(row):
    
    def convert_point(point):
        return f"({point[0]}, {point[1]}, '{point[2]}')"
    
    def convert_path(path_list):
        return "[" + ", ".join([f"({p[0]}, {p[1]})" for p in path_list]) + "]"
    
    # def convert_sub_path(sub_path_list):
    #     return "[" + ", ".join([f"(({sp[0][0]}, {sp[0][1]}), ({sp[1][0]}, {sp[1][1]}), {sp[2]})" for sp in sub_path_list]) + "]"
    
    # start_point, target_point, cur_point 등을 문자열로 변환
    row['start_point'] = convert_point(row['start_point'])
    row['target_point'] = convert_point(row['target_point'])
    row['cur_point'] = convert_point(row['cur_point'])
    # path_TimeMachine 및 sub_path_time_TimeMachine 등을 문자열로 변환
    row['path_TimeMachine'] = convert_path(row['path_TimeMachine'])
    print(row['sub_path_time_TimeMachine'])
    # row['sub_path_time_TimeMachine'] = convert_sub_path(row['sub_path_time_TimeMachine'])
 
    # sub_path_time_LastAPI를 문자열로 변환
    # row['sub_path_time_LastAPI'] = convert_sub_path(row['sub_path_time_LastAPI'])
    return row



class AgentMovement:
    def __init__(self, initial_lat, initial_lon, speed=1.2, interval=1, radius=100, random_speed=False):
        self.lat = initial_lat  # 현재 위도
        self.lon = initial_lon  # 현재 경도
        self.base_speed = speed  # 기본 이동 속도
        self.interval = interval  # 시간 간격
        self.radius = radius  # 반경 (기준점에서 100m)
        self.random_speed = random_speed  # 랜덤 속도 여부
        self.ref_lat = initial_lat  # 기준점 위도
        self.ref_lon = initial_lon  # 기준점 경도
        self.path = [(initial_lat, initial_lon)]  # 이동 경로 저장
        self.circles = [(initial_lat, initial_lon, radius)]  # 반경을 벗어났을 때의 기준점 저장
        self.steps = 0  # 이동 횟수 (step 카운터)
        self.dis=0

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        # 위도와 경도를 라디안으로 변환
        lat1 = math.radians(lat1)
        lon1 = math.radians(lon1)
        lat2 = math.radians(lat2)
        lon2 = math.radians(lon2)
        
        # 지구의 반지름 (미터 단위)
        R = 6378137
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # 하버사인 공식을 사용하여 두 지점 간의 거리 계산
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance

    def move_randomly(self,intervals,bearing):
        # 랜덤한 방위각(0에서 2π 사이)
        # bearing = random.uniform(0, 2 * math.pi)
        
        # 이동 속도를 랜덤으로 설정하거나 고정 속도를 사용
        if self.random_speed:
            speed = self.base_speed*random.uniform(0.5, 1.5)  # 0.5m/s에서 3.0m/s 사이의 랜덤 속도
        else:
            speed = self.base_speed
        
        # 이동 거리 = 속도 * 시간 간격
        distance = speed * intervals

        # 현재 위도와 경도를 라디안으로 변환
        lat1 = math.radians(self.lat)
        lon1 = math.radians(self.lon)

        # 지구의 반지름 (미터 단위)
        R = 6378137
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

        return new_lat, new_lon, speed

    def move_one_step(self, intervals=1,bearing=0):
        # 한 번의 이동을 수행하는 메서드 (한 스텝)
        new_lat, new_lon, speed = self.move_randomly(intervals,bearing)
        
        # 현재 기준점과 새로운 위치 사이의 거리 계산
        distance_from_ref = self.calculate_distance(self.ref_lat, self.ref_lon, new_lat, new_lon)
        # print(f"현재 위치: {new_lat}, {new_lon}, 속도: {speed:.2f}m/s, 기준점과의 거리: {distance_from_ref:.2f}m")
        self.dis=0
        # 기준점으로부터 100m를 벗어났는지 확인
        if distance_from_ref > self.radius:
            # print(f"#####100m를 벗어남! 새로운 기준점 설정: {new_lat}, {new_lon}#####")
            self.ref_lat, self.ref_lon = new_lat, new_lon
            # 기준점을 벗어났을 때 새로운 기준점 저장
            self.circles.append((new_lat, new_lon, self.radius))
            self.dis=1

        # 새로운 위치로 현재 위치 갱신
        self.lat, self.lon = new_lat, new_lon
        self.path.append((self.lat, self.lon))  # 이동 경로에 추가
        self.steps += 1  # 스텝 카운터 증가

        # 상태 정보를 반환 (필요한 경우)
        return {
            "lat": self.lat,
            "lon": self.lon,
            "speed": speed,
            "distance_from_ref": distance_from_ref,
            "steps": self.steps,
            "ref_lat": self.ref_lat,
            "ref_lon": self.ref_lon,
            "over_100m" : self.dis
        }


# 시간과 호출 횟수를 관리하는 클래스
class ContextBuilder:
    def __init__(self, appKey):
        self.tmap_appKey = appKey[0]
        self.odsay_appKey = appKey[1]
        
        # self.initialize()  # 초기화 메서드 호출

    def initialize(self,start_point=None,target_point=None,target_time=None):
        
        self.min_diff = timedelta.max
        self.path_time_ans = 0
        # time.sleep(2*60-1)
        self.round = 0 # 초기 설정
        self.leaving_time = None
        self.context_history = pd.DataFrame()
        odsay_cur_time = datetime.now(tz) # 현재시간 저장
        r1, r2 = sample_region('work')
        r3, r4 = sample_region('live')
        
        if start_point == None:
            # self.start_point = {"name": get_poi_names(self.tmap_appKey)} #random start point 설정
            # # self.start_point = {"name": '포시즌마트'} #random start point 설정
            # get_lon_lat(self.start_point, self.tmap_appKey)
            self.start_point = {}
         
            res1 = get_poi_names(self.tmap_appKey,region=r3,option='live')
       
            self.start_point["name"] = res1[0]
            self.start_point["lat"] = res1[1]
            self.start_point["lon"] = res1[2]
        else:
            self.start_point=start_point
        
        self.current_point = self.start_point.copy()
        
        if target_point == None:
            # self.target_point = {"name": get_poi_names(self.tmap_appKey)}
            # get_lon_lat(self.target_point, self.tmap_appKey)
            # # self.target_point = {"name": 'CU 트라팰리스점'}
            self.target_point = {}
            res2 = get_poi_names(self.tmap_appKey,region=r1,option='work')
            self.target_point["name"] = res2[0]
            self.target_point["lat"] = res2[1]
            self.target_point["lon"] = res2[2]
        else:
            self.target_point=target_point
        
        
        if target_time == None:        
          
            
            start=(float(self.start_point["lat"]), float(self.start_point["lon"]), 0)
            target=(float(self.target_point["lat"]), float(self.target_point["lon"]), 0)
            
            # 타겟 타임 설정을 위한 실시간 쿼리
            query_rt = query_realtime_odsay(self.start_point, self.target_point, self.odsay_appKey)
     
            duration = query_to_duration_odsay(query_rt)
            print(duration)
            if not isinstance(duration, int) or duration < 20:
                print('초기화')
                self.initialize()  # 다시 초기화 시도
                return
            first_cur_time = datetime.now(tz)
            self.target_time = max(first_cur_time + timedelta(minutes=duration * 1.5), first_cur_time + timedelta(minutes=16))
        else:
            self.target_time = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%S%z")

        query_tm = query_realtime_odsay(self.current_point, self.target_point, self.odsay_appKey)
        self.id_hashed = generate_id(self.target_point, self.target_time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        self.start_time = odsay_cur_time
        self.duration_tm = query_to_duration_odsay(query_tm)
        self.path_TimeMachine = query_to_path_odsay(query_tm)
        self.path_LastAPI = self.path_TimeMachine
        self.sub_path_time_TimeMachine = query_to_sub_path_time_odsay(query_tm)
        self.call_time_TimeMachine = self.target_time - timedelta(minutes=self.duration_tm)
        self.req_leaving_time = self.target_time - timedelta(minutes=self.duration_tm + 8)
        '''
        id, start_time,target_time,start_point,target_point,cur_time,cur_point,path_time_TimeMachine
        '''
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
                    "transportation": "public",
                    "round": self.round,
                    "start_time": self.start_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "target_time": self.target_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "start_point": (float(self.start_point["lat"]), float(self.start_point["lon"]), 0),
                    "target_point": (float(self.target_point["lat"]), float(self.target_point["lon"]), 0),
                    "call_time_TimeMachine": None, #check
                    "path_TimeMachine": self.path_TimeMachine,
                    "sub_path_time_TimeMachine": self.sub_path_time_TimeMachine,
                    "path_time_TimeMachine": self.duration_tm,
                    "cur_time": self.start_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "cur_point": (float(self.current_point["lat"]), float(self.current_point["lon"]), 0),
                    "call_time_LastAPI": None,
                    "call_point_LastAPI": None,
                    "path_LastAPI": None,
                    "path_time_LastAPI": None,
                    "req_leaving_time": self.req_leaving_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "sub_path_time_LastAPI":None, #check
                    "weather": np.nan,
                    "path_time_ans": 0,
                    "event": np.nan,
                }
            ]
        )
        
        self.df = pd.concat([self.df, initial_row], ignore_index=True)
        return self.df

    def new_context(self, query_rt):
        print('context_input')

 
        print('odsay')
        path_LastAPI =  query_to_path_odsay(query_rt)
        path_time_LastAPI = query_to_duration_odsay(query_rt)
        sub_path_time_LastAPI = query_to_sub_path_time_odsay(query_rt)


        new_row = pd.DataFrame(
                [
                    {
                        "id": self.id_hashed,
                        "group": "realtime",
                        "transportation": "public",
                        "round": self.round,
                        "start_time": self.start_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "target_time": self.target_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "start_point": (float(self.start_point["lat"]), float(self.start_point["lon"]), 0),
                        "target_point": (float(self.target_point["lat"]), float(self.target_point["lon"]), 0),
                        "call_time_TimeMachine": self.call_time_TimeMachine.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "path_TimeMachine": self.path_TimeMachine,
                        "sub_path_time_TimeMachine": self.sub_path_time_TimeMachine,
                        "path_time_TimeMachine": self.duration_tm,
                        "cur_time": datetime.now(self.target_time.tzinfo).strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "cur_point": (float(self.current_point["lat"]), float(self.current_point["lon"]), 0 ),
                        "call_time_LastAPI": datetime.now(self.target_time.tzinfo).strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "call_point_LastAPI": (float(self.current_point["lat"]), float(self.current_point["lon"]), 0),
                        "path_LastAPI": path_LastAPI,
                        "path_time_LastAPI": path_time_LastAPI,
                        "req_leaving_time": self.req_leaving_time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "sub_path_time_LastAPI": sub_path_time_LastAPI,
                        "weather": None,
                        "path_time_ans": 0,
                        "event": None,
                    }
                ]
            )
        # new_row['start_point'] = new_row['start_point'].apply(lambda x: str(x))
        # new_row['target_time'] = new_row['target_time'].apply(lambda x: str(x))
        # new_row['cur_point'] = new_row['cur_point'].apply(lambda x: str(x))
        # new_row['call_time_LastAPI'] = new_row['call_time_LastAPI'].apply(lambda x: str(x))
        return new_row
 
    
#     # model에 context를 전달
# appKeys =  [('jICW6XpKkB9hSdzIJrySU4EtcJ7JzDTZBBPDeHo7', 'bfeae872ac9a35817ff18366aeb852dd')] 
# # run_time_checker(appKeys[0])
# a = ContextBuilder(appKeys[0])

# # context random으로 받기
# data_api = a.initialize()
# data_api['cur_point'][0][1]
# data_api['target_point'][0][0]    

if __name__ == "__main__":
    '''
    출근시간:8시-10시 도착
    퇴근시간:16시-17시 출발
    '''
    while (True):
        try:
            # current_time = datetime.now()
            # current_hour = current_time.hour

            # # 활성 시간대: 오전 6시~10시 또는 오후 16시~18시
            # if (6 <= current_hour < 10) or (16 <= current_hour < 18):
            #     pass  # 활성 시간대에서는 바로 실행
            # else:
            #     # 다음 활성 시간대까지 남은 시간 계산
            #     if current_hour < 6:
            #         # 현재 시간이 6시 이전인 경우
            #         next_active_time = current_time.replace(hour=6, minute=0, second=0, microsecond=0)
            #     elif 10 <= current_hour < 16:
            #         # 현재 시간이 오전 10시~오후 4시 사이인 경우
            #         next_active_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
            #     elif current_hour >= 18:
            #         # 현재 시간이 오후 6시 이후인 경우
            #         next_active_time = (current_time + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)

            #     # 남은 시간 계산
            #     remaining_seconds = (next_active_time - current_time).total_seconds()

            #     # 최대 300초(5분) 동안 sleep, 활성 시간대에 가까워질수록 sleep 시간 줄어듦
            #     sleep_time = min(remaining_seconds, 3000)
            #     time.sleep(sleep_time)
            bearing = random.uniform(0, 2 * math.pi)
            speed_value=random.randint(0, 4)
            # speed_value=100
            print(f'speed: {speed_value}')
            
 #####################################공통###########################################
            # appKeys[0][0]:tmap , appKeys[0][1]:odsay
            # appKeys =  [('jICW6XpKkB9hSdzIJrySU4EtcJ7JzDTZBBPDeHo7', 'U09vONCdcNfuvgottMHMkDMv/1s7hiA/Il/KwCj5hw0')]
            # appKeys =  [('g0f0ehnV4P9y8j57TJ4UpaC1krpED1807RzMZqRm', 'S+8tbonZfWp6lIp0Oonw/+L0RdmYToA7hI2l2jAkLVg')]
            appKeys =  [('TJjxwqXsxI5y0e3wE96qc4ZGFHJpZxmwNUqkYrra', 'S+8tbonZfWp6lIp0Oonw/+L0RdmYToA7hI2l2jAkLVg')]
            
            # run_time_checker(appKeys[0])
            cb = ContextBuilder(appKeys[0])
            id = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')
            # context random으로 받기
            print('odsay1')
            data_api_odsay = cb.initialize().apply(convert_to_second_format, axis=1)
            print('odsay_init1')
            cur_time = datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S%z")

            output = {}
            output['id'] = id
            output['contexts'] = {}
            output['status'] = 0        # 0: 진행, 1: 출발에 의한 종료, 2: New Episode에 의한 종료
            output['descriptions'] = None

            episode=0
#####################################분리###########################################
            # our model
            print('odsay_init2')
            
            public_prediction = PublicPrediction()
            # rulebased
            # baronow_rulebased = BaroNowRuleBased()

            print('odsay_init3')
            data_api_tmap = data_api_odsay.copy()
            # data_api['transportation'] = 'car_kakao'
            data_api_tmap['transportation'] = 'car_tmap'

            
            api_call_time_odsay = None
            leaving_time_odsay = None
            api_call_time_tmap = None
            leaving_time_tmap = None
            
            data_history_odsay = pd.DataFrame()
            data_history_tmap = pd.DataFrame()
            
            new_episodes_odsay = None
            new_episodes_tmap = None
            
            new_context_odsay=1
            new_context_tmap=1
            
            round_odsay = 0
            round_tmap = 0
            
            output_odsay = output.copy()
            output_tmap = output.copy()
            
            api_count_odsay = 1
            api_count_tmap = 1

            previous_time =datetime.now(tz)
            cur_point={}
            target_point={}
############################움직이는 agent##############################################################################################################################
            start_lat, start_lon, *rest = ast.literal_eval(data_api_odsay['start_point'][0])
            agent = AgentMovement(initial_lat=start_lat, initial_lon= start_lon, speed=int(speed_value), interval=2, radius=100, random_speed=True)
######################################################################################################################################################################
            while(True):
                cur_time = datetime.now(tz)
                
                # stop 시그널
                if (data_api_odsay['req_leaving_time'][0] < data_api_odsay['start_time'][0]) and (episode ==0):
                    print('next episode')
                    break
                
#########################확인용 저장#######################################################################################################################################
                if episode ==0:
                    csv_file='/home/baronow/BaroNowProject_V3/online_result/odsay/odsay_status0.csv'
                    # 파일이 이미 존재하는지 확인
                    if os.path.exists(csv_file):
                        # 기존 파일에 데이터 추가 (header=False로 기존 헤더를 덮어쓰지 않음)
                        data_api_odsay.to_csv(csv_file, mode='a', header=False, index=False)              
                    else:
                        # 파일이 없으면 새로 생성
                        data_api_odsay.to_csv(csv_file, mode='w', header=True, index=False)
                episode=1
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
                # (API Call 시간 도래시) odsay
                if output_odsay['status'] != 1:
                    
                    # 초기 data 행
                    if round_odsay <1:
                        context_odsay = data_api_odsay.copy()
                        
                        context_odsay['round']=0
                        data_history_odsay = pd.concat([data_history_odsay, context_odsay], axis=0)
                        round_odsay += 1
                    # ---------------------------------------------------------------------------------------
                    # (NewEpisode Time 도래시)
                    if (new_episodes_odsay is not None) and (
                        (leaving_time_odsay is None and cur_time > datetime.strptime(new_episodes_odsay, "%Y-%m-%dT%H:%M:%S%z"))
                        or (leaving_time_odsay is not None and datetime.strptime(leaving_time_odsay,"%Y-%m-%dT%H:%M:%S%z") > datetime.strptime(new_episodes_odsay,"%Y-%m-%dT%H:%M:%S%z")) 
                        ):
                            new_context_odsay=1
                            output_odsay['status'] = 2     # 종료조건 인자
                            output_odsay['descriptions'] = "New_Episodes"
                    # ---------------------------------------------------------------------------------------
                    
                    # ---------------------------------------------------------------------------------------
                    # (Leaving Time 도래시)
                    elif (leaving_time_odsay is not None) and (cur_time > datetime.strptime(leaving_time_odsay, "%Y-%m-%dT%H:%M:%S%z")):
                        new_context_odsay=1
                        #############################################
                        last_context_odsay = pd.DataFrame(data_history_odsay.iloc[-1,:]).T.copy()
                        print('hi')
                        # print(data_history_odsay['cur_point'])
                        # print(last_context_odsay['cur_point'])
                        lat_cur, lon_cur , name_cur = ast.literal_eval(last_context_odsay['cur_point'].iloc[0])
                        lat_target, lon_target, name_target = ast.literal_eval(last_context_odsay['target_point'].iloc[0])
                        cur_point["name"] = name_cur
                        cur_point["lat"] = lat_cur
                        cur_point["lon"] = lon_cur

                        target_point["name"] = name_target
                        target_point["lat"] = lat_target
                        target_point["lon"] = lon_target

                        query_rt_odsay = query_realtime_odsay(cur_point, target_point, appKeys[0][1]) 
                        print('odsay2')
                        context_odsay = cb.new_context(query_rt=query_rt_odsay)
                        print('odsay3')
                        context_odsay['round']=round_odsay
                        data_history_odsay = pd.concat([data_history_odsay, context_odsay], axis=0)
                        
                        output_odsay['status'] = 1     # 종료조건 인자
                        output_odsay['descriptions'] = "Leaving to Destination. End_of_episodes."

                else:
                    pass
                    # --------------------------------------------------------------------------------------- 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
                            # --------------------------------------------------------------------------------------- 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
                # odsay
                if output_odsay['status'] !=1 and new_context_odsay == 1:
        
                    # print('new1')
                    # output['contexts'] = data_history.to_dict()
                    output_odsay['contexts'] = data_history_odsay.to_dict('records')
                    # print('ouput_context')
                    # for key, value in output['contexts'][0].items():
                    #     print(f"{key}: {type(value)}")
                    ########################################################################
                    # Preprocessing
                    context_df = pd.DataFrame(output_odsay['contexts'])
                    context_df['start_point'] = context_df['start_point'].apply(lambda x: str(x))
                    context_df['target_point'] = context_df['target_point'].apply(lambda x: str(x))
                    context_df['cur_point'] = context_df['cur_point'].apply(lambda x: str(x))
                    context_df['call_point_LastAPI'] = context_df['call_point_LastAPI'].apply(lambda x: str(x))
                    context_df = context_df.applymap(lambda x: None if x =='None' else x)

                    for api_col in ['call_time_LastAPI', 'call_point_LastAPI', 'path_LastAPI', 'path_time_LastAPI']:
                        if api_col in ['call_point_LastAPI', 'path_LastAPI']:
                            context_df[api_col] = context_df[api_col].apply(lambda x: str(x))
                        
                        # if len(context_df) > 0:
                        #     # context.at[context.index[0], api_col] = context_df.iloc[-1,:][api_col]
                        #     context_df[api_col] = context_df.iloc[-1,:][api_col]
                        # else:
                        #     context_df[api_col] = np.nan
                    output_odsay['contexts'] = context_df.to_dict('records')
                    ########################################################################
                    print('new2')
                    response_pred_odsay = public_prediction.predict(output_odsay, mode='online', return_full_info=True)
                    ########################################################################
                    # print('res')
                    # odsay
                    # ---------------------------------------------------------------------------------------
                    if 'api_call_time' in response_pred_odsay.keys() and response_pred_odsay['api_call_time'] is not None:
                        api_call_time_odsay = response_pred_odsay['api_call_time']

                    # 출발시간 정보
                    if 'leaving_time' in response_pred_odsay.keys() and response_pred_odsay['leaving_time'] is not None:
                        leaving_time_odsay = response_pred_odsay['leaving_time']

                    # 위치이동에 따른 기존 Episode 종료 및 새로운 Episode시작
                    if 'new_episodes' in response_pred_odsay.keys() and response_pred_odsay['new_episodes'] is not None:
                        new_episodes_odsay = response_pred_odsay['new_episodes']
                    # ---------------------------------------------------------------------------------------
                    
                    new_context_odsay=0
                    # print log
                    print(f"odsay 정보: {cur_time}  {output_odsay['descriptions']} {response_pred_odsay}")
                    
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
                
#agent move ------------------------------------------------------------------------------------------------------------------------------------------------------------
                # agent move
                current_time = datetime.now(tz)
                time_difference = current_time - previous_time
                state = agent.move_one_step(intervals=time_difference.total_seconds(),bearing=bearing)
                previous_time=current_time

                if state['over_100m'] == 1:
                    if output_odsay['status'] != 1:
                        new_context_odsay=1
                        print(f"Step: {state['steps']}, 위치: ({state['lat']}, {state['lon']}), 속도: {state['speed']:.2f}, 기준점과의 거리: {state['distance_from_ref']:.2f}m")
                        moving_new_context = pd.DataFrame(data_history_odsay.iloc[-1, :]).T.copy()
                        moving_new_context.at[moving_new_context.index[0], 'cur_point'] = (state['lat'], state['lon'], '0')
                        moving_new_context['cur_point']=moving_new_context['cur_point'].apply(lambda x: str(x))
                        moving_new_context['round']=round_odsay

                        moving_new_context['cur_time']=datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S%z")
                        
                        data_history_odsay = pd.concat([data_history_odsay, moving_new_context], axis=0, ignore_index=True)
                        round_odsay += 1
                    else:
                        pass
                    
# ---------------------------------------------------------------------------------------
                # Episode 종료
                if output_odsay['status'] == 1:   # 출발
                    
                    ##############확인용##############
                    df = pd.DataFrame(output_odsay)
                    csv_file='/home/baronow/BaroNowProject_V3/online_result/odsay/output_odsay.csv'
                    # 파일이 이미 존재하는지 확인
                    if os.path.exists(csv_file):
                        # 기존 파일에 데이터 추가 (header=False로 기존 헤더를 덮어쓰지 않음)
                        df.to_csv(csv_file, mode='a', header=False, index=False)
                    else:
                        # 파일이 없으면 새로 생성
                        df.to_csv(csv_file, mode='w', header=True, index=False)
                    ##############확인용##############
                
                    print('end!!')
                    break
# ---------------------------------------------------------------------------------------                
                if output_odsay['status'] == 2:  # 새로운 에피소드로 계산
                    ##############확인용##############
                    df = pd.DataFrame(output_odsay)
                    csv_file='/home/baronow/BaroNowProject_V3/online_result/odsay/output_odsay.csv'
                    # 파일이 이미 존재하는지 확인
                    if os.path.exists(csv_file):
                        # 기존 파일에 데이터 추가 (header=False로 기존 헤더를 덮어쓰지 않음)
                        df.to_csv(csv_file, mode='a', header=False, index=False)
                    else:
                        # 파일이 없으면 새로 생성
                        df.to_csv(csv_file, mode='w', header=True, index=False)
                    ##############확인용##############
                    print('odsay new episode start')

                    output_odsay = {}
                    output_odsay['id'] = id
                    output_odsay['contexts'] = {}
                    output_odsay['status'] = 0        # 0: 진행, 1: 출발에 의한 종료, 2: New Episode에 의한 종료
                    output_odsay['descriptions'] = None
                    latest_info = pd.DataFrame(data_history_odsay.iloc[[-1], :]).reset_index(drop=True).copy()

                    print(latest_info['target_point'])
                    print(latest_info['target_point'][0])
                    print(type(latest_info['target_point'][0]))
                    print(type(str(latest_info['target_point'][0])))
                    latest_info['target_point']=latest_info['target_point'].apply(lambda x: str(x))
                    latest_info['cur_point']=latest_info['cur_point'].apply(lambda x: str(x))
                    
                    lat, lon, name = eval(latest_info['target_point'][0])
                    print(type(lat))
                    print(type(lon))
             
                    # 딕셔너리로 변환
                    target_point = {
                        'lat': lat,
                        'lon': lon,
                        'name': '0'
                    }
                    print('this3')
                    lat, lon, name = eval(latest_info['cur_point'][0])
                    print('this4')
                    # 딕셔너리로 변환
                    cur_point = {
                        'lat': lat,
                        'lon': lon,
                        'name': '0'
                    }

                    print(cur_point)
                    data_history_odsay = pd.DataFrame()
                    round_odsay = 0
                    data_api_odsay = cb.initialize(start_point=cur_point,target_point=target_point,target_time=latest_info['target_time'][0]).apply(convert_to_second_format, axis=1)
                    # data_api.to_csv('/home/baronow/BaroNowProject_V3/move.csv',index=False)
                    print(target_point)
                    api_count_odsay += 1
                    #################확인용#####################
                    csv_file='/home/baronow/BaroNowProject_V3/online_result/odsay/odsay_status2.csv'
                    # 파일이 이미 존재하는지 확인
                    if os.path.exists(csv_file):
                        # 기존 파일에 데이터 추가 (header=False로 기존 헤더를 덮어쓰지 않음)
                        data_api_odsay.to_csv(csv_file, mode='a', header=False, index=False)
                    else:
                        # 파일이 없으면 새로 생성
                        data_api_odsay.to_csv(csv_file, mode='w', header=True, index=False)
                    ##########################################
                else:
                    pass
# --------------------------------------------------------------------------------------- 

            if output_odsay['status'] ==1:
                print('end episode')
                # odsay Log 쌓기위한 전달
                
                output_odsay['contexts'] = data_history_odsay.to_dict('records')
                context_df = pd.DataFrame(output_odsay['contexts'])
                context_df['start_point'] = context_df['start_point'].apply(lambda x: str(x))
                context_df['target_point'] = context_df['target_point'].apply(lambda x: str(x))
                context_df['cur_point'] = context_df['cur_point'].apply(lambda x: str(x))
                context_df['call_point_LastAPI'] = context_df['call_point_LastAPI'].apply(lambda x: str(x))
                context_df = context_df.applymap(lambda x: None if x =='None' else x)

                for api_col in ['call_time_LastAPI', 'call_point_LastAPI', 'path_LastAPI', 'path_time_LastAPI']:
                    if api_col in ['call_point_LastAPI', 'path_LastAPI']:
                        context_df[api_col] = context_df[api_col].apply(lambda x: str(x))
                output_odsay['contexts'] = context_df.to_dict('records')  
                #++++++++++++++++++++++++++++++++++++++++++++++++#
                public_prediction.predict(output_odsay, mode='online', return_full_info=True)
                #++++++++++++++++++++++++++++++++++++++++++++++++#
                    
                # Performance Evaluation odsay----------------------------------------------------------------------------------------
                last_context_odsay = pd.DataFrame(data_history_odsay).iloc[-1,:]
                print('test1')
                if not pd.isna(last_context_odsay['path_time_LastAPI']):
                    print('test2')
                    # print(last_context_odsay['path_time_LastAPI'])
                    arrival_time = datetime.strptime(last_context_odsay['cur_time'], "%Y-%m-%dT%H:%M:%S%z") + timedelta(minutes=int(last_context_odsay['path_time_LastAPI']))
                    print('test3')
                    print(arrival_time)
                    residual_arrival_time = (arrival_time - datetime.strptime(last_context_odsay['target_time'], "%Y-%m-%dT%H:%M:%S%z") ).total_seconds()/60
                    print(f"Is in allowance range? : {-13 < residual_arrival_time < 0}, {residual_arrival_time}")
                    with open("/home/baronow/BaroNowProject_V3/online_result/result_odsay.txt", "a") as file:  # 'a' 모드는 파일에 덧붙여 쓰기
                        file.write(f"{id}-odsay: avg_Velocity: {speed_value}m/s, # of api call: {api_count_odsay}, Is in allowance range? : {-13 < residual_arrival_time < 0}, {residual_arrival_time}\n")
                
##############################################  
            # break
        except KeyError as e:
            print(f"KeyError 발생: {e}. 초기화 시도 중...")
            # 예외 발생 시 초기화 시도
            continue

        except Exception as e:
            print(f"알 수 없는 에러 발생: {e}. 초기화 시도 중...")
            # 다른 예외 발생 시 초기화 시도
            continue


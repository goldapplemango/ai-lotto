
#feature_utils4.py

import os
import sys
import pandas as pd
import json

import numpy as np

COLUMN_NAMES = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']
RANGES = {
    '1-10': range(1, 11),
    '11-20': range(11, 21),
    '21-30': range(21, 31),
    '31-40': range(31, 41),
    '41-50': range(41, 51),
}

# ---- Code-5: 피처 엔지니어링 ----
def dynamic_feature_engineering(model, X_train, X_test, importance_threshold=0.05):
    """
    피처 엔지니어링 및 중요도에 따라 피처 제거
    """ # 모델을 사용하여 피처 중요도 평가
    # model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 중요도가 낮은 피처를 제거
    importances = model.feature_importances_
    important_features = [i for i, imp in enumerate(importances) if imp > importance_threshold]
    
    X_train = X_train[:, important_features]
    X_test = X_test[:, important_features]
    
    return X_train, X_test

def generate_features(df):
#    df = pd.DataFrame()

    print("feature 데이터 크기 출력 ; ", df.shape)  # 데이터 크기 출력
    print("데이터 컬럼 출력 : ", df.columns)  # 데이터 컬럼 출력
#    print(df.head())
    print("COLUMN_NAMES 확인:", COLUMN_NAMES)
    print("statistics 이전 : ", df[COLUMN_NAMES].head())  # 컬럼 데이터 확인

    # 통계적 특징
    statistics = calculate_statistics(df)
    # 홀수/짝수 개수
    parity = calculate_parity(df)
    # 연속번호 여부
    consecutive = calculate_consecutive(df)
    # 번호 간 간격
    gaps = calculate_gaps(df)
    # 구간별 번호 분포
    range_distribution = calculate_range_distribution(df, RANGES)
    # 핫 넘버 분석
    hot_cold = calculate_hot_cold_numbers(df)
    # 회차 기반 특징
    sequence_features = add_sequence_features(df)
    # 번호 등장 빈도
    frequency_features = calculate_frequency_of_appearance(df)
    # 모든 feature 통합
    features = pd.concat([statistics, parity, consecutive, gaps, range_distribution, hot_cold, sequence_features, frequency_features], axis=1)
    # 최종 데이터에 병합
    result = pd.concat([df, features], axis=1)
    return result

# 통계적 특징 계산
def calculate_statistics(df):
    features = pd.DataFrame()
    try:
        features['합계'] = df[COLUMN_NAMES].sum(axis=1)
        print("합계 생성 완료")
        features['평균값'] = df[COLUMN_NAMES].mean(axis=1)
        print("평균값 생성 완료")
        features['표준편차'] = df[COLUMN_NAMES].std(axis=1)
        print("표준편차 생성 완료")
        features['최소값'] = df[COLUMN_NAMES].min(axis=1)
        print("최소값 생성 완료")
        features['최대값'] = df[COLUMN_NAMES].max(axis=1)
        print("최대값 생성 완료")
        features['범위'] = features['최대값'] - features['최소값']
        print("범위 생성 완료")
    except Exception as e:
        print(f"calculate_statistics 오류: {e}")
    return features


# 홀수/짝수 개수 계산
def calculate_parity(df):
    features = pd.DataFrame()
    features['홀수개수'] = df[COLUMN_NAMES].apply(lambda row: sum(num % 2 != 0 for num in row), axis=1)
    features['짝수개수'] = len(COLUMN_NAMES) - features['홀수개수']
    return features

# 연속 번호 여부 계산
def calculate_consecutive(df):
    def has_consecutive(numbers):
        sorted_numbers = sorted(numbers)
        for i in range(len(sorted_numbers) - 1):
            if sorted_numbers[i + 1] - sorted_numbers[i] == 1:
                return 1
        return 0
    features = pd.DataFrame()
    features['연속번호'] = df[COLUMN_NAMES].apply(lambda row: has_consecutive(row), axis=1)
    return features

# 번호 간 간격 계산
def calculate_gaps(df):
    def calculate_row_gaps(numbers):
        sorted_numbers = sorted(numbers)
        return [sorted_numbers[i + 1] - sorted_numbers[i] for i in range(len(sorted_numbers) - 1)]
    gaps = df[COLUMN_NAMES].apply(lambda row: calculate_row_gaps(row), axis=1, result_type='expand')
    gaps.columns = ['간격1', '간격2', '간격3', '간격4', '간격5']
    return gaps

# 번호가 각 구간에 얼마나 속하는지 계산
def calculate_range_distribution(df, ranges):
    features = pd.DataFrame()
    for range_name, number_range in ranges.items():
        features[f'{range_name}_개수'] = df[COLUMN_NAMES].apply(
            lambda row: sum(num in number_range for num in row), axis=1
        )
    return features

# 핫 넘버 분석
def calculate_hot_cold_numbers(df):
    features = pd.DataFrame()
    all_numbers = df[COLUMN_NAMES].values.flatten()
    unique_numbers, counts = np.unique(all_numbers, return_counts=True)
    number_count = dict(zip(unique_numbers, counts))
    features['핫넘버'] = df[COLUMN_NAMES].apply(lambda row: sum(number_count.get(num, 0) for num in row), axis=1)
    return features

# 회차 기반 추가 특징 생성
def add_sequence_features(df):
    features = pd.DataFrame()
    features['이전합계차'] = df['합계'].diff().fillna(0)
    return features

# 등장 빈도 계산
def calculate_frequency_of_appearance(df):
    features = pd.DataFrame()
    all_numbers = df[COLUMN_NAMES].values.flatten()
    unique_numbers, counts = np.unique(all_numbers, return_counts=True)
    frequency_dict = dict(zip(unique_numbers, counts))
    for number in range(1, 51):
        features[f'{number}_빈도'] = df[COLUMN_NAMES].apply(lambda row: row.tolist().count(number), axis=1)
    return features

#    feature_utils.py  $#$$
def save_important_features(features, path):
    """중요 피처 저장"""
    with open(path, 'w') as file:
        json.dump(features, file)
    print(f"중요 피처 저장 완료: {path}")

def load_important_features(path):
    """중요 피처 불러오기"""
    if (path):
        with open(path, 'r') as file:
            features = json.load(file)
        print(f"중요 피처 불러오기 완료: {path}")
        return features
    else:
        print("중요 피처 파일이 없습니다. 새로 생성합니다.")
        return None

#




# ai-main12.ipynb

import subprocess
import os
import sys
import pandas as pd

from google.colab import drive

def mount_drive():
    try:
        # 이미 마운트된 경우 예외 처리
        if not os.path.isdir('/content/drive'):
            print("Google Drive를 마운트 중입니다...")
            drive.mount('/content/drive')
            print("Google Drive 마운트 완료!")
        else:
            print("Google Drive가 이미 마운트되어 있습니다.")
    except Exception as e:
        print(f"Google Drive 마운트 실패: {e}")
        sys.exit(1)

# 호출 예시
mount_drive()

def install_libraries(libraries):
    for lib in libraries:
        try:
            # 라이브러리 임포트를 시도
            __import__(lib)
            print(f"{lib} 라이브러리가 이미 설치되어 있습니다.")
        except ImportError:
            # 설치되지 않은 경우 설치 진행
            print(f"{lib} 라이브러리를 설치합니다...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"{lib} 설치가 완료되었습니다.")

# 필수 라이브러리 목록
required_libraries = [ 'numpy', 'optuna', 'pandas','import-ipynb'] # scikit-learn

# 동적 설치 함수 실행
install_libraries(required_libraries)

# !pip install import-ipynb

# lib 설치후 sys.path에 추가
sys.path.append("/content/drive/MyDrive/lotto4")

import time

import import_ipynb
import numpy as np

from utils4 import conditional_tuning, log_progress, provide_recommendations
from model_utils4 import (get_model_path, train_individual_models, train_meta_model,
 save_model, load_model, evaluate_model, optimize_hyperparameters)
from feature_utils4 import generate_features10


def load_and_preprocess_data(data_path):
    """데이터 로드 및 전처리"""
    try:
        print(f"데이터 경로: {data_path}")
        data = pd.read_csv(data_path)
        print("데이터 로드 성공")
        print(data.head())  # 데이터 일부 출력

        # 결측값 확인
        if data.isnull().any().any():
            raise ValueError("데이터에 결측치가 있습니다.")

        # 필요한 컬럼만 선택
        required_columns = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6', '보너스']
        if not set(required_columns).issubset(data.columns):
            raise ValueError(f"필요한 컬럼이 없습니다: {required_columns}")
        data = data[required_columns]  # 필요한 컬럼만 선택

        # print(data.head())  #

        # 피처 생성
        data = generate_features10(data)
        # print(data.head())  #

        return data
    except Exception as e:
        print(f"데이터 처리 오류: {e}")
        return None


# 타이머 및 조건 변수
last_tuning_time = time.time()
tuning_delay = 7 * 24 * 3600  # 7일 (단위: 초)
epoch = 0


def main():

    # 경로 설정
    act_path = "/content/drive/MyDrive/lotto4"
    data_path = f"{act_path}/lotto_data11.csv"
    print(f"데이터 경로: {data_path} 입니다.")


    individual_model_path_rf = get_model_path("trained_model_individual_rf", act_path)
    individual_model_path_gb = get_model_path("trained_model_individual_gb", act_path)
    meta_model_path = get_model_path("trained_model_meta", act_path)

    print("로또 분석 프로그램 시작")

    # 데이터 로드 및 피처 생성
    data = load_and_preprocess_data(data_path)
    print(data.shape)  # 데이터 크기 출력
    print(data.columns)  # 데이터 컬럼 출력

    # 이후 코드 유지...

    print("로또 분석 프로그램 시작")

    # 데이터 로드 및 피처 생성
    data = load_and_preprocess_data(data_path)

    if data is None:
        print("데이터 로딩에 실패했습니다. 프로그램을 종료합니다.")
        return
        # print(data.shape)  # 데이터 크기 출력
        # print(data.columns)  # 데이터 컬럼 출력

    # 데이터 분할
    X = data.drop(columns=['보너스']).values
    y = data['보너스'].values

    # 모델 로드 또는 학습
    individual_models = load_model(individual_model_path_gb, act_path)
    meta_model = load_model(meta_model_path, act_path)

    if individual_models is None or meta_model is None:
        print("기존 모델이 없어 새로 학습을 시작합니다...")
        individual_models = train_individual_models(X, y)
        meta_model = train_meta_model(individual_models, X, y)
        save_model(individual_models, individual_model_path_rf, act_path)
        save_model(meta_model, meta_model_path, act_path)

    # 학습 루프 시작
    best_eval_accuracy = 0
    last_tuning_time = time.time()

    for epoch in range(1, 1001):
        print(f"\n{epoch}회 학습 시작...")

        # 모델 학습 후 정확도 계산
        current_model, train_accuracy = train_individual_models(X, y)  # 모델 학습
        eval_accuracy = evaluate_model(current_model, X, y)  # 모델 평가

        # 최고 평가 정확도 갱신
        if eval_accuracy > best_eval_accuracy:
           best_eval_accuracy = eval_accuracy  # 최고 성능 갱신
           save_model(current_model, f"{act_path}/best_model.pkl", act_path)  # 모델 저장

        # 조건부 하이퍼파라미터 튜닝
        best_params, last_tuning_time = conditional_tuning(epoch, eval_accuracy, best_eval_accuracy, last_tuning_time, X, y)

        if best_params:
            current_model.set_params(**best_params)
            current_model.fit(X, y)
            print("튜닝 후 모델 재학습 완료.")

        log_progress(epoch, best_eval_accuracy)
        time.sleep(5)  # 학습 간격 설정

    # 추천 번호 제공
    provide_recommendations(current_model, X)
    print("프로그램을 종료합니다.")

if __name__ == "__main__":
    main()

# end main
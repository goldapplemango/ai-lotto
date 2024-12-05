#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 4utils.py

import json
import time
import os
import sys #sys

from google.colab import drive
drive.mount("/content/drive")

import numpy as np

def remove_special(act_path, filename):

    # 코드 내 모든 non-breaking space를 일반 공백으로 교체
    with open(f"{act_path}/{filename}", 'r', encoding='utf-8') as file:
        content = file.read()
        content = content.replace('\u00A0', ' ')  # 불연속 공백을 일반 공백으로 변환

    with open(f"{act_path}/{filename}", 'w', encoding='utf-8') as file:
        file.write(content)
    return None

# !jupyter nbconvert --to python /content/drive/MyDrive/lotto4/model_utils.ipynb

# act_path = "/content/drive/MyDrive/lotto4"

# filename = "feature_utils4.py"
# filename1 = "model_utils4.py"
# filename2 = "utils4.py"
# filename3 = "ai-main1202.ipynb"

# remove_special(act_path, filename)
# remove_special(act_path, filename1)
# remove_special(act_path, filename2)
# remove_special(act_path, filename3)

def log_progress(epoch, best_score):
    """학습 진행 상황 로그 기록"""
    with open("progress.log", "a") as log_file:
        log_file.write(f"Epoch {epoch}: Best Accuracy: {best_score}\n")

# 타이머 및 조건 변수

def provide_recommendations(models, data, num_sets=5):
    """로또 번호 추천 (1~45 범위로 조정)"""
    recommendations = []
    for _ in range(num_sets):
        sample = data.sample(n=1)
        predictions = sum([model.predict(sample) for model in models]) / len(models)
        numbers = np.round(predictions).astype(int)
        numbers = np.clip(numbers, 1, 45)  # 1~45 범위로 조정
        numbers = np.unique(numbers)  # 중복 제거
        if len(numbers) < 6:  # 6개 번호가 안되면 추가
            additional_numbers = np.random.choice(range(1, 46), 6 - len(numbers), replace=False)
            numbers = np.concatenate([numbers, additional_numbers])
        recommendations.append(sorted(numbers[:6]))  # 6개 번호 추천
    for idx, rec in enumerate(recommendations, 1):
        print(f"세트 {idx}: 번호: {rec}")


# end


# In[ ]:


last_tuning_time = time.time()
tuning_delay = 7 * 24 * 3600  # 7일 (단위: 초)

def conditional_tuning(epoch, current_accuracy, previous_accuracy, X_train, y_train, last_tuning_time):
    """
    조건부 및 기간 기반 하이퍼파라미터 튜닝 함수.

    Parameters:
        epoch (int): 현재 학습 epoch.
        current_accuracy (float): 현재 모델 정확도.
        previous_accuracy (float): 이전 모델 정확도.
        new_data_length (int): 현재 데이터 길이.
        last_data_length (int): 이전 데이터 길이.
        X_train (DataFrame): 학습 데이터.
        y_train (Series): 학습 라벨.

    Returns:
        dict: 최적화된 하이퍼파라미터 (튜닝 시).
        None: 튜닝 조건 미충족 시.
    """

    # 조건 1: 성능 저하 감지
    performance_drop = (current_accuracy - previous_accuracy) < 0.01  # 정확도 1% 이하 상승
    # 조건 2: 데이터 증가 감지
    data_growth = new_data_length > (last_data_length * 1.1)  # 데이터가 10% 이상 증가

    # 조건부 튜닝 실행
    if performance_drop or data_growth:
        print(f"[조건 기반 튜닝] 조건 변화 감지 (Epoch {epoch})")
        best_params = optimize_hyperparameters(X_train, y_train)
        last_tuning_time = time.time()  # 타이머 리셋
        return best_params, last_tuning_time

    # 기간 기반 튜닝 실행
    if time.time() - last_tuning_time > tuning_delay:
        print(f"[기간 기반 튜닝] 타이머 경과 (Epoch {epoch})")
        best_params = optimize_hyperparameters(X_train, y_train)
        last_tuning_time = time.time()  # 타이머 리셋
        return best_params,last_tuning_time

    return None  # 튜닝 조건 미충족 시 None 반환


def train_and_predict_loop(data, epochs=100, prediction_interval=10):
    best_model = None
    best_score = 0

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}] Training started...")
        
        # Step 1: Train the model
        models = train_individual_models(data["X"], data["y"])
        meta_model = train_meta_model(models, data["X"], data["y"])
        
        # Step 2: Evaluate the model
        score = evaluate_model(models, data["X"], data["y"])
        if score > best_score:
            best_score = score
            best_model = models
        
        # Step 3: Predict and provide recommendations
        if epoch % prediction_interval == 0:
            print(f"[Prediction Interval] Generating recommendations at epoch {epoch}...")
            predictions = provide_recommendations(best_model, data["X"])
            print(f"Predictions: {predictions}")
        
        # Step 4: Adaptive learning with feedback
        feedback = get_user_feedback()  # Placeholder for user interaction
        if feedback:
            data = augment_data_with_feedback(data, feedback)

        # Step 5: Save progress
        log_progress(epoch, best_score)

    return best_model


def strategic_warfare_ai(data, epochs=100, prediction_interval=10):
    best_model = None
    best_score = 0

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}] Training started...")
        
        # Step 1: Train models
        models = train_individual_models(data["X"], data["y"])
        meta_model = train_meta_model(models, data["X"], data["y"])
        
        # Step 2: Evaluate performance
        score = evaluate_model(models, data["X"], data["y"])
        if score > best_score:
            best_score = score
            best_model = models
        
        # Step 3: Predict dynamically
        if epoch % prediction_interval == 0:
            print(f"[Prediction Interval] Epoch {epoch}: Generating predictions...")
            predictions = provide_recommendations(best_model, data["X"])
            print(f"Predictions: {predictions}")
        
        # Step 4: Self-improvement
        feedback = gather_feedback(predictions)  # Placeholder for user interaction
        if feedback:
            data = integrate_feedback(data, feedback)

        # Step 5: Adaptive optimization
        if should_tune(epoch, score, best_score):
            print("[Adaptive Tuning] Performance stagnation detected. Optimizing...")
            tune_hyperparameters(data["X"], data["y"])

        log_progress(epoch, best_score)

    return best_model
def strategic_warfare_ai(data, epochs=100, prediction_interval=10):
    best_model = None
    best_score = 0

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}] Training started...")
        
        # Step 1: Train models
        models = train_individual_models(data["X"], data["y"])
        meta_model = train_meta_model(models, data["X"], data["y"])
        
        # Step 2: Evaluate performance
        score = evaluate_model(models, data["X"], data["y"])
        if score > best_score:
            best_score = score
            best_model = models
        
        # Step 3: Predict dynamically
        if epoch % prediction_interval == 0:
            print(f"[Prediction Interval] Epoch {epoch}: Generating predictions...")
            predictions = provide_recommendations(best_model, data["X"])
            print(f"Predictions: {predictions}")
        
        # Step 4: Self-improvement
        feedback = gather_feedback(predictions)  # Placeholder for user interaction
        if feedback:
            data = integrate_feedback(data, feedback)

        # Step 5: Adaptive optimization
        if should_tune(epoch, score, best_score):
            print("[Adaptive Tuning] Performance stagnation detected. Optimizing...")
            tune_hyperparameters(data["X"], data["y"])

        log_progress(epoch, best_score)

    return best_model

def train_and_predict_loop(data, epochs=100, prediction_interval=10):
    """
    학습과 예측이 통합된 메인 루프
    """
    best_model = None
    best_score = 0
    feedback_log = []

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}] Training started...")
        
        # Step 1: 학습 진행
        models = train_individual_models(data["X"], data["y"])
        meta_model = train_meta_model(models, data["X"], data["y"])
        
        # Step 2: 성능 평가
        score = evaluate_model(models, data["X"], data["y"])
        if score > best_score:
            best_score = score
            best_model = models

        # Step 3: 예측 생성 및 사용자 피드백
        if epoch % prediction_interval == 0:
            print(f"[Prediction Interval] Epoch {epoch}: Generating predictions...")
            predictions = provide_recommendations(best_model, data["X"])
            print(f"Predictions: {predictions}")
            feedback = gather_feedback(predictions)
            if feedback:
                feedback_log.append(feedback)
                data = integrate_feedback(data, feedback)

        # Step 4: 적응형 학습
        if should_tune(epoch, score, best_score):
            print("[Adaptive Tuning] Performance stagnation detected. Optimizing...")
            tune_hyperparameters(data["X"], data["y"])

        # Step 5: 진행 상황 로깅
        log_progress(epoch, best_score)

    return best_model, feedback_log

def provide_recommendations(models, data, num_sets=5):
    """
    다중 전략에 기반한 로또 번호 추천
    """
    strategies = ["stable", "risky", "mixed"]
    recommendations = {}

    for strategy in strategies:
        strategy_recommendations = []
        for _ in range(num_sets):
            sample = data.sample(n=1)
            predictions = sum([model.predict(sample) for model in models]) / len(models)
            numbers = np.round(predictions).astype(int)
            numbers = np.clip(numbers, 1, 45)  # 번호 범위를 1~45로 제한
            numbers = np.unique(numbers)  # 중복 제거

            if len(numbers) < 6:  # 번호가 6개 미만일 경우 추가 번호 채움
                additional_numbers = np.random.choice(range(1, 46), 6 - len(numbers), replace=False)
                numbers = np.concatenate([numbers, additional_numbers])
            
            strategy_recommendations.append(sorted(numbers[:6]))
        
        recommendations[strategy] = strategy_recommendations
        print(f"\n[{strategy.upper()} STRATEGY] Recommendations:")
        for idx, rec in enumerate(strategy_recommendations, 1):
            print(f"Set {idx}: {rec}")
    
    return recommendations


def integrate_feedback(data, feedback):
    """
    사용자 피드백을 학습 데이터에 통합
    """
    feedback_data = pd.DataFrame(feedback, columns=data["X"].columns)
    feedback_labels = feedback["labels"]
    new_data = pd.concat([data["X"], feedback_data])
    new_labels = pd.concat([data["y"], feedback_labels])
    
    print("Feedback successfully integrated into the training dataset.")
    return {"X": new_data, "y": new_labels}

def analyze_human_patterns(data, user_choices):
    """
    인간 군집 패턴을 학습하여 적합하지 않은 조합을 제거하거나 조정
    """
    # 과거 데이터에서 군집 행동 분석
    common_patterns = data["X"].apply(lambda x: tuple(sorted(x)), axis=1).value_counts()
    unpopular_combinations = common_patterns[common_patterns < common_patterns.median()].index

    # 사용자 선택 데이터와 교차 비교
    adjusted_choices = []
    for choice in user_choices:
        if tuple(sorted(choice)) in unpopular_combinations:
            # 덜 흔한 조합으로 교체
            alternative = np.random.choice(range(1, 46), 6, replace=False)
            adjusted_choices.append(sorted(alternative))
        else:
            adjusted_choices.append(sorted(choice))
    return adjusted_choices


def analyze_human_patterns(data, user_choices):
    """
    인간 군집 패턴을 학습하여 적합하지 않은 조합을 제거하거나 조정
    """
    # 과거 데이터에서 군집 행동 분석
    common_patterns = data["X"].apply(lambda x: tuple(sorted(x)), axis=1).value_counts()
    unpopular_combinations = common_patterns[common_patterns < common_patterns.median()].index

    # 사용자 선택 데이터와 교차 비교
    adjusted_choices = []
    for choice in user_choices:
        if tuple(sorted(choice)) in unpopular_combinations:
            # 덜 흔한 조합으로 교체
            alternative = np.random.choice(range(1, 46), 6, replace=False)
            adjusted_choices.append(sorted(alternative))
        else:
            adjusted_choices.append(sorted(choice))
    return adjusted_choices


def analyze_machine_bias(data):
    """
    1, 2, 3호기 추첨 데이터에서 패턴을 학습
    """
    # 기계별로 데이터를 분리
    machine_data = data.groupby("machine_number")

    patterns = {}
    for machine, group in machine_data:
        print(f"Analyzing machine {machine}...")
        # 특정 기계에서 자주 등장한 번호 찾기
        common_numbers = group.iloc[:, :6].stack().value_counts()
        patterns[machine] = common_numbers[common_numbers > common_numbers.median()].index.tolist()
    
    return patterns


def integrate_failure_analysis(data, failed_predictions):
    """
    실패 데이터를 학습에 통합하여 불필요한 조합을 제거
    """
    failed_patterns = pd.DataFrame(failed_predictions).apply(lambda x: tuple(sorted(x)), axis=1)
    failed_counts = failed_patterns.value_counts()

    # 실패 패턴 제거
    data["X"] = data["X"].apply(lambda x: x if tuple(sorted(x)) not in failed_counts else np.random.choice(range(1, 46), 6, replace=False))
    return data


def preprocess_data(raw_data, machine_data=None, user_choices=None):
    """
    데이터를 전처리하고 군집, 기계, 사용자 정보를 통합
    """
    # 기본 전처리
    processed_data = generate_features(raw_data)

    # 기계별 데이터 패턴 추가
    if machine_data is not None:
        machine_patterns = analyze_machine_bias(machine_data)
        processed_data = merge_machine_patterns(processed_data, machine_patterns)

    # 사용자 입력 데이터 처리
    if user_choices is not None:
        processed_data["user_bias"] = analyze_human_patterns(processed_data, user_choices)
    
    return processed_data


def train_predict_feedback_loop(data, epochs=100, prediction_interval=10):
    """
    학습과 예측을 통합한 메인 루프
    """
    best_model = None
    best_score = 0
    failed_predictions = []

    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}] Training started...")

        # Step 1: 모델 학습
        models = train_individual_models(data["X"], data["y"])
        meta_model = train_meta_model(models, data["X"], data["y"])
        
        # Step 2: 성능 평가
        score = evaluate_model(models, data["X"], data["y"])
        if score > best_score:
            best_score = score
            best_model = models

        # Step 3: 예측 생성
        if epoch % prediction_interval == 0:
            print(f"[Prediction Interval] Generating predictions...")
            predictions = provide_recommendations(best_model, data["X"])
            print(f"Predictions: {predictions}")

            # 실패 데이터 기록
            failed_predictions.extend(predictions)

        # Step 4: 실패 데이터 통합
        if failed_predictions:
            data = integrate_failure_analysis(data, failed_predictions)
        
        # Step 5: 하이퍼파라미터 튜닝
        if should_tune(epoch, score, best_score):
            print("[Adaptive Tuning] Optimizing hyperparameters...")
            tune_hyperparameters(data["X"], data["y"])

        log_progress(epoch, best_score)

    return best_model


def user_interface(model, data):
    """
    사용자와의 상호작용 인터페이스
    """
    # 추천 번호 제공
    print("\n--- Recommended Numbers ---")
    strategies = provide_recommendations(model, data["X"])
    
    # 전략별 결과 표시
    for strategy, recommendations in strategies.items():
        print(f"\n[{strategy.upper()} STRATEGY]")
        for idx, rec in enumerate(recommendations, 1):
            print(f"Set {idx}: {rec}")

    # 사용자의 선택 기록
    user_feedback = input("Choose your favorite set (or press Enter to skip): ")
    if user_feedback:
        feedback = process_user_feedback(user_feedback, strategies)
        return feedback
    return None


import pandas as pd

def load_lotto_data(file_path):
    """
    로또 데이터를 로드하고 기본적인 전처리
    """
    try:
        data = pd.read_csv(file_path)
        print(f"데이터 로드 성공: {file_path}")
        print(data.head())  # 데이터 샘플 출력
        return data
    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return None

# 파일 경로 설정
file_path = "/content/drive/MyDrive/lotto4/lotto_data.csv"
lotto_data = load_lotto_data(file_path)


def preprocess_lotto_data(data):
    """
    로또 데이터 전처리
    """
    # 필요한 컬럼만 선택 (번호1~번호6, 보너스)
    required_columns = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6', '보너스']
    data = data[required_columns]
    
    # 결측치 확인
    if data.isnull().any().any():
        print("결측치 존재, 제거 중...")
        data = data.dropna()  # 결측치 제거
    
    # 번호 데이터를 1~45로 제한
    for i in range(1, 7):
        data[f'번호{i}'] = data[f'번호{i}'].clip(1, 45)
    
    # 보너스 번호도 1~45로 제한
    data['보너스'] = data['보너스'].clip(1, 45)

    return data

lotto_data = preprocess_lotto_data(lotto_data)


def analyze_number_frequency(data):
    """
    로또 번호의 등장 빈도 분석
    """
    all_numbers = data[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].values.flatten()
    unique_numbers, counts = np.unique(all_numbers, return_counts=True)
    frequency_dict = dict(zip(unique_numbers, counts))

    return frequency_dict

number_frequency = analyze_number_frequency(lotto_data)
print(f"로또 번호 등장 빈도: {number_frequency}")


import matplotlib.pyplot as plt

def plot_number_frequency(frequency_dict):
    """
    로또 번호의 등장 빈도 시각화
    """
    numbers = list(frequency_dict.keys())
    counts = list(frequency_dict.values())

    plt.bar(numbers, counts)
    plt.xlabel("번호")
    plt.ylabel("빈도수")
    plt.title("로또 번호 등장 빈도")
    plt.show()

plot_number_frequency(number_frequency)



def generate_features(data):
    """
    로또 데이터에서 특징을 생성
    """
    features = pd.DataFrame()
    features['합계'] = data[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].sum(axis=1)
    features['평균'] = data[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].mean(axis=1)
    features['표준편차'] = data[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].std(axis=1)

    return features

features = generate_features(lotto_data)


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_models(features, target):
    """
    로또 번호 예측 모델 학습
    """
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)

    rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)
    gb_rmse = mean_squared_error(y_test, gb_pred, squared=False)

    print(f"랜덤 포레스트 모델 RMSE: {rf_rmse:.4f}")
    print(f"그라디언트 부스팅 모델 RMSE: {gb_rmse:.4f}")

    return rf_model, gb_model

target = lotto_data['보너스']  # 예측 목표는 보너스 번호
models = train_models(features, target)



def provide_recommendations(models, features, num_sets=5):
    """
    예측 번호 추천
    """
    recommendations = []
    for _ in range(num_sets):
        # 예측 번호 생성
        rf_pred = models[0].predict(features)
        gb_pred = models[1].predict(features)

        # 두 모델의 예측 평균 계산
        prediction = (rf_pred + gb_pred) / 2
        recommendation = np.clip(np.round(prediction), 1, 45)
        recommendations.append(recommendation)

    return recommendations

# 예측 생성
recommendations = provide_recommendations(models, features)
print(f"추천 번호 세트: {recommendations}")

def integrate_feedback(data, feedback):
    """
    피드백 데이터를 학습 데이터에 통합
    """
    feedback_data = pd.DataFrame(feedback, columns=data.columns)
    updated_data = pd.concat([data, feedback_data], axis=0)
    return updated_data

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import numpy as np

def train_models(features, target):
    """
    여러 모델을 학습시키고, 그 예측을 비교하여 가장 성능 좋은 모델을 선택
    """
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # 각 모델 정의
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

    # 학습
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # 예측
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    # 모델 평가
    rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)
    gb_rmse = mean_squared_error(y_test, gb_pred, squared=False)
    xgb_rmse = mean_squared_error(y_test, xgb_pred, squared=False)

    print(f"랜덤 포레스트 모델 RMSE: {rf_rmse:.4f}")
    print(f"그라디언트 부스팅 모델 RMSE: {gb_rmse:.4f}")
    print(f"XGBoost 모델 RMSE: {xgb_rmse:.4f}")

    # 모델의 예측 성능에 따라 가중치를 부여하여 앙상블 예측 수행
    model_rmse = np.array([rf_rmse, gb_rmse, xgb_rmse])
    weights = 1 / (model_rmse + 1e-5)  # RMSE가 낮을수록 높은 가중치 부여
    weights = weights / weights.sum()  # 가중치 정규화

    # 앙상블 예측: 가중 평균 방식
    ensemble_pred = (rf_pred * weights[0] + gb_pred * weights[1] + xgb_pred * weights[2])

    print(f"앙상블 예측 RMSE: {mean_squared_error(y_test, ensemble_pred, squared=False):.4f}")

    return rf_model, gb_model, xgb_model, ensemble_pred



def provide_ensemble_recommendations(models, features, num_sets=5):
    """
    앙상블 모델을 통해 로또 번호 추천
    """
    rf_model, gb_model, xgb_model, _ = models

    recommendations = []
    for _ in range(num_sets):
        # 각 모델에서 예측값을 얻고
        rf_pred = rf_model.predict(features)
        gb_pred = gb_model.predict(features)
        xgb_pred = xgb_model.predict(features)

        # 예측값의 평균을 구하여 추천 번호 생성
        ensemble_pred = (rf_pred + gb_pred + xgb_pred) / 3

        # 1~45 범위로 번호를 제한
        recommendation = np.clip(np.round(ensemble_pred), 1, 45)
        recommendations.append(recommendation)

    return recommendations



def generate_features(data):
    """
    로또 번호에서 다양한 통계적 특성 생성
    """
    features = pd.DataFrame()

    # 번호들의 합계와 평균 계산
    features['합계'] = data[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].sum(axis=1)
    features['평균'] = data[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].mean(axis=1)

    # 각 회차마다 번호의 표준편차
    features['표준편차'] = data[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].std(axis=1)

    # 번호들 간의 차이(간격) 계산
    def calculate_gaps(row):
        sorted_row = sorted(row)
        gaps = [sorted_row[i+1] - sorted_row[i] for i in range(len(sorted_row)-1)]
        return gaps

    features['간격'] = data[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].apply(calculate_gaps, axis=1)

    return features

# 특성 생성 적용
features = generate_features(lotto_data)


def prepare_data(data):
    """
    데이터에서 특징과 목표 변수 준비
    """
    # 특징(features)과 목표 변수(target) 설정
    X = generate_features(data)  # 특성 생성
    y = data['보너스']  # 목표 변수 (보너스 번호)
    
    return X, y

X, y = prepare_data(lotto_data)


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import numpy as np

def train_ensemble_models(X, y):
    """
    앙상블 모델 학습 및 예측
    """
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 정의
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

    # 학습
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # 예측
    rf_pred = rf_model.predict(X_test)
    gb_pred = gb_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    # 모델 평가 (RMSE)
    rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)
    gb_rmse = mean_squared_error(y_test, gb_pred, squared=False)
    xgb_rmse = mean_squared_error(y_test, xgb_pred, squared=False)

    print(f"랜덤 포레스트 모델 RMSE: {rf_rmse:.4f}")
    print(f"그라디언트 부스팅 모델 RMSE: {gb_rmse:.4f}")
    print(f"XGBoost 모델 RMSE: {xgb_rmse:.4f}")

    # 앙상블 모델: 가중 평균 예측
    model_rmse = np.array([rf_rmse, gb_rmse, xgb_rmse])
    weights = 1 / (model_rmse + 1e-5)  # RMSE가 낮을수록 높은 가중치 부여
    weights = weights / weights.sum()  # 가중치 정규화

    # 앙상블 예측: 가중 평균 방식
    ensemble_pred = (rf_pred * weights[0] + gb_pred * weights[1] + xgb_pred * weights[2])

    print(f"앙상블 예측 RMSE: {mean_squared_error(y_test, ensemble_pred, squared=False):.4f}")

    return rf_model, gb_model, xgb_model, ensemble_pred

# 모델 학습 및 예측
models = train_ensemble_models(X, y)

def provide_ensemble_recommendations(models, X, num_sets=5):
    """
    앙상블 모델을 통해 로또 번호 추천
    """
    rf_model, gb_model, xgb_model, _ = models

    recommendations = []
    for _ in range(num_sets):
        # 각 모델에서 예측값을 얻고
        rf_pred = rf_model.predict(X)
        gb_pred = gb_model.predict(X)
        xgb_pred = xgb_model.predict(X)

        # 예측값의 평균을 구하여 추천 번호 생성
        ensemble_pred = (rf_pred + gb_pred + xgb_pred) / 3

        # 1~45 범위로 번호를 제한
        recommendation = np.clip(np.round(ensemble_pred), 1, 45)
        recommendations.append(recommendation)

    return recommendations

def integrate_feedback(data, feedback):
    """
    피드백 데이터를 학습 데이터에 통합
    """
    feedback_data = pd.DataFrame(feedback, columns=data.columns)
    updated_data = pd.concat([data, feedback_data], axis=0)
    return updated_data


def integrate_feedback_and_retrain(models, data, feedback, X, y):
    """
    피드백을 학습 데이터에 통합하고 모델을 재학습시키는 함수
    """
    # 피드백을 데이터에 통합
    feedback_data = pd.DataFrame(feedback, columns=data.columns)
    updated_data = pd.concat([data, feedback_data], axis=0)

    # 모델 재학습
    X_updated, y_updated = prepare_data(updated_data)
    
    # 각 모델 학습
    rf_model, gb_model, xgb_model, _ = train_ensemble_models(X_updated, y_updated)
    
    return rf_model, gb_model, xgb_model


def provide_recommendations(models, X, num_sets=5):
    """
    앙상블 모델을 통해 로또 번호 추천
    """
    rf_model, gb_model, xgb_model, _ = models
    recommendations = []

    for _ in range(num_sets):
        # 각 모델에서 예측값을 얻고
        rf_pred = rf_model.predict(X)
        gb_pred = gb_model.predict(X)
        xgb_pred = xgb_model.predict(X)

        # 예측값의 평균을 구하여 추천 번호 생성
        ensemble_pred = (rf_pred + gb_pred + xgb_pred) / 3

        # 1~45 범위로 번호를 제한
        recommendation = np.clip(np.round(ensemble_pred), 1, 45)
        recommendations.append(recommendation)

    return recommendations



def get_user_feedback():
    """
    사용자에게 번호를 선택하게 하고, 그 선택을 피드백으로 반환
    """
    feedback = input("추천 번호 중 선택하세요 (comma로 구분): ")
    feedback = list(map(int, feedback.split(',')))
    return feedback

# 예시 사용법
user_feedback = get_user_feedback()  # 사용자가 선택한 번호를 입력받음


import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def optimize_hyperparameters(X_train, y_train):
    """
    Optuna를 이용한 하이퍼파라미터 최적화
    """
    def objective(trial):
        # 모델 하이퍼파라미터 정의
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_train)
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    return study.best_params


def evaluate_performance(predictions, actual_values):
    """
    모델의 예측 성능을 평가하는 함수
    :param predictions: 모델의 예측 값
    :param actual_values: 실제 로또 당첨 번호
    :return: 정확도 및 성공률
    """
    correct_predictions = sum([1 if pred in actual_values else 0 for pred in predictions])
    accuracy = correct_predictions / len(predictions)
    
    success_rate = sum([1 if any(pred == actual for pred in predictions) for actual in actual_values]) / len(actual_values)
    
    print(f"정확도: {accuracy * 100:.2f}%")
    print(f"성공률: {success_rate * 100:.2f}%")
    return accuracy, success_rate


def log_performance(epoch, accuracy, success_rate):
    """
    성능 로그를 기록하고 저장하는 함수
    :param epoch: 현재 학습의 epoch 번호
    :param accuracy: 모델의 정확도
    :param success_rate: 모델의 성공률
    """
    with open("performance_log.txt", "a") as log_file:
        log_file.write(f"Epoch {epoch}: 정확도 = {accuracy:.2f}%, 성공률 = {success_rate:.2f}%\n")
    
    print(f"Epoch {epoch}: 성능 기록 완료!")


def update_model_with_new_data(models, new_data, new_feedback):
    """
    새로운 데이터와 피드백을 모델에 반영하여 재학습하는 함수
    :param models: 기존의 학습된 모델들 (앙상블)
    :param new_data: 새로운 데이터
    :param new_feedback: 사용자의 새로운 피드백
    :return: 업데이트된 모델
    """
    # 피드백 통합
    updated_data = integrate_feedback_and_retrain(models, new_data, new_feedback, new_data[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']], new_data['보너스'])

    return updated_data

def real_time_prediction_and_feedback(models, data, num_sets=5):
    """
    실시간 예측 생성 및 사용자 피드백을 통한 모델 개선
    """
    # 예측 번호 생성
    recommendations = provide_ensemble_recommendations(models, data[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']], num_sets)
    
    print("\n추천 번호 세트:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"세트 {idx}: {rec}")

    # 사용자 피드백 받기
    user_feedback = get_user_feedback()  # 사용자 피드백 받기
    
    # 모델 업데이트
    updated_models = update_model_with_new_data(models, data, user_feedback)

    return updated_models

def simulate_lotto_round(models, data, actual_numbers, num_sets=5):
    """
    로또 추첨 시뮬레이션 (추천된 번호와 실제 번호 비교)
    """
    # 추천 번호 생성
    recommendations = provide_ensemble_recommendations(models, data[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']], num_sets)
    
    # 실제 번호와 추천 번호 비교
    for i, rec in enumerate(recommendations, 1):
        accuracy, success_rate = evaluate_performance(rec, actual_numbers)
        log_performance(i, accuracy, success_rate)
    
    return recommendations


import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def optimize_hyperparameters(X_train, y_train):
    """
    Optuna를 이용한 하이퍼파라미터 최적화
    """
    def objective(trial):
        # 모델 하이퍼파라미터 정의
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_train)
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    return study.best_params


import os
import pickle

def save_model_checkpoint(model, epoch, file_path="model_checkpoint.pkl"):
    """
    모델 체크포인트 저장
    """
    checkpoint = {
        'model': model,
        'epoch': epoch
    }
    with open(file_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Model checkpoint saved at epoch {epoch}.")

def load_model_checkpoint(file_path="model_checkpoint.pkl"):
    """
    모델 체크포인트 로드
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Model checkpoint loaded from epoch {checkpoint['epoch']}.")
        return checkpoint['model'], checkpoint['epoch']
    else:
        print("No checkpoint found. Starting from scratch.")
        return None, 0

def save_training_progress(epoch, accuracy, success_rate, log_file="training_progress.log"):
    """
    훈련 상태를 로그 파일에 기록
    """
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch}: Accuracy={accuracy:.2f}%, Success Rate={success_rate:.2f}%\n")
    print(f"Training progress saved at epoch {epoch}.")

def load_training_progress(log_file="training_progress.log"):
    """
    훈련 로그를 로드하여 마지막 상태를 반환
    """
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        last_epoch = int(lines[-1].split()[1][:-1])  # 마지막 에포크 번호 추출
        print(f"Training progress loaded from epoch {last_epoch}.")
        return last_epoch
    else:
        print("No training progress found. Starting from scratch.")
        return 0

def save_results_and_state(epoch, results, file_path="results_checkpoint.pkl"):
    """
    학습 결과와 상태를 저장
    """
    checkpoint = {
        'epoch': epoch,
        'results': results
    }
    with open(file_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Results and state saved at epoch {epoch}.")

def load_results_and_state(file_path="results_checkpoint.pkl"):
    """
    학습 결과와 상태를 로드
    """
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"Results and state loaded from epoch {checkpoint['epoch']}.")
        return checkpoint['epoch'], checkpoint['results']
    else:
        print("No results checkpoint found. Starting fresh.")
        return 0, None

import shutil
from datetime import datetime

def backup_system_state(state_folder="model_backups"):
    """
    시스템 상태 백업
    """
    if not os.path.exists(state_folder):
        os.makedirs(state_folder)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(state_folder, f"backup_{timestamp}")
    
    # 모델 및 결과 파일 복사
    shutil.copy("model_checkpoint.pkl", backup_path)
    shutil.copy("training_progress.log", backup_path)
    shutil.copy("results_checkpoint.pkl", backup_path)
    
    print(f"Backup created at {backup_path}.")


import logging

def setup_logging(log_file="system_log.log"):
    """
    로그 파일 설정
    """
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info("System started.")
    print("Logging initialized.")

def log_message(message, log_file="system_log.log"):
    """
    로그 메시지 기록
    """
    logging.info(message)
    print(f"Logged: {message}")


def auto_recovery_and_resume():
    """
    시스템 중단 후 자동으로 복구하고 학습을 재개하는 함수
    """
    # 체크포인트 로드
    model, last_epoch = load_model_checkpoint()
    last_epoch = load_training_progress()
    results = load_results_and_state()

    if model is None:
        print("No checkpoint found. Starting fresh.")
        # 모델 새로 학습
        model = initialize_new_model()  # 새로운 모델 초기화
    
    # 복구된 상태에서 학습 재개
    for epoch in range(last_epoch, total_epochs):
        # 모델 학습 및 예측
        model = train_model(model, epoch)
        # 예측 결과 저장 및 평가
        results = evaluate_and_save_results(model, epoch)

        # 체크포인트 저장
        save_model_checkpoint(model, epoch)
        save_training_progress(epoch, accuracy, success_rate)
        save_results_and_state(epoch, results)

    print("System resumed and training completed.")


import optuna
from sklearn.metrics import mean_squared_error

def optimize_hyperparameters(X_train, y_train):
    """
    Optuna를 이용한 하이퍼파라미터 최적화
    """
    def objective(trial):
        # 모델 하이퍼파라미터 정의
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_train)
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    return study.best_params


def augment_data_with_real_time_data(historical_data, real_time_data):
    """
    과거 데이터와 실시간 데이터를 결합하여 학습 데이터 증강
    """
    augmented_data = pd.concat([historical_data, real_time_data], axis=0)
    return augmented_data


def generate_synthetic_data(data, num_samples=1000):
    """
    가상 로또 번호 데이터를 생성하여 학습 데이터 증강
    """
    synthetic_data = []
    for _ in range(num_samples):
        synthetic_sample = np.random.choice(range(1, 46), size=6, replace=False)
        synthetic_data.append(synthetic_sample)
    
    synthetic_df = pd.DataFrame(synthetic_data, columns=['번호1', '번호2', '번호3', '번호4', '번호5', '번호6'])
    return synthetic_df

def integrate_user_feedback(predictions, actual, feedback_data):
    """
    사용자 피드백을 통합하여 학습 데이터를 갱신하고 모델 개선
    """
    # 예측과 실제 번호의 차이를 분석
    feedback = {"predicted": predictions, "actual": actual, "feedback": feedback_data}
    
    # 피드백에 기반하여 데이터를 갱신
    updated_data = pd.DataFrame(feedback)
    return updated_data


def update_model_with_feedback(model, data, feedback, X, y):
    """
    피드백을 모델에 반영하여 실시간으로 학습하는 함수
    """
    # 피드백을 모델에 통합
    updated_data = integrate_user_feedback(model.predict(X), y, feedback)

    # 모델 재학습
    updated_model = train_ensemble_models(X, updated_data['target'])
    
    return updated_model

def track_performance(epoch, accuracy, success_rate, log_file="performance_log.txt"):
    """
    성과를 추적하고 기록하는 시스템
    """
    with open(log_file, 'a') as f:
        f.write(f"Epoch {epoch}: Accuracy={accuracy:.2f}%, Success Rate={success_rate:.2f}%\n")
    
    print(f"Performance logged at epoch {epoch}.")


def evaluate_and_adjust_model(models, X_test, y_test):
    """
    주기적으로 모델 성능을 평가하고 조정
    """
    predictions = [model.predict(X_test) for model in models]
    
    # 성과 평가
    accuracy = evaluate_performance(predictions, y_test)
    
    # 성과가 좋지 않으면 하이퍼파라미터 조정
    if accuracy < 0.7:
        print("Model performance is low. Adjusting parameters...")
        new_params = optimize_hyperparameters(X_test, y_test)
        models = train_ensemble_models(X_test, y_test, new_params)
    
    return models


def scheduled_batch_learning():
    """
    주기적으로 배치 학습을 실행하여 모델 개선
    """
    while True:
        # 주기적으로 모델을 재학습
        models = train_ensemble_models(X_train, y_train)
        track_performance(epoch, accuracy, success_rate)
        
        # 일주일 간격으로 모델 재학습
        time.sleep(7 * 24 * 60 * 60)  # 1주일 대기


def optimize_system_performance():
    """
    모델 성능을 지속적으로 최적화하는 함수
    """
    # 기존 모델을 최적화하여 성능을 향상
    optimized_models = optimize_hyperparameters(X_train, y_train)
    
    # 모델 성능 평가 후 재학습
    models = evaluate_and_adjust_model(optimized_models, X_test, y_test)
    
    return models


def select_best_model(models, X_test, y_test):
    """
    여러 모델을 성능 기준으로 평가하고, 가장 성능이 좋은 모델을 선택
    """
    best_model = None
    best_score = float('inf')  # 가장 낮은 RMSE를 찾기 위해 초기값 설정
    for model in models:
        pred = model.predict(X_test)
        score = mean_squared_error(y_test, pred, squared=False)  # RMSE 계산
        if score < best_score:
            best_score = score
            best_model = model
    print(f"Best model selected with RMSE: {best_score:.4f}")
    return best_model


def auto_tune_model(model, X_train, y_train):
    """
    모델의 하이퍼파라미터를 자동으로 튜닝하는 함수
    """
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'learning_rate': [0.05, 0.1, 0.2]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    tuned_model = grid_search.best_estimator_
    return tuned_model

import optuna
from sklearn.ensemble import RandomForestRegressor

def optimize_hyperparameters_with_optuna(X_train, y_train):
    """
    Optuna를 이용한 하이퍼파라미터 최적화
    """
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 200)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_train)
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        return rmse
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print(f"Best hyperparameters: {study.best_params}")
    return study.best_params


def integrate_user_feedback(model, feedback_data, X, y):
    """
    사용자 피드백을 기반으로 실시간으로 모델을 학습하는 함수
    """
    # 피드백 데이터를 반영하여 업데이트
    new_data = pd.DataFrame(feedback_data)
    X_updated, y_updated = prepare_data(new_data)

    # 모델 재학습
    model.fit(X_updated, y_updated)
    return model


def log_and_monitor_performance(predictions, actual_values, epoch, log_file="performance_log.txt"):
    """
    실시간 예측 결과를 기록하고, 성과를 추적하는 함수
    """
    accuracy, success_rate = evaluate_performance(predictions, actual_values)
    save_training_progress(epoch, accuracy, success_rate, log_file)
    print(f"Epoch {epoch}: Accuracy={accuracy:.2f}%, Success Rate={success_rate:.2f}%")


import time

def periodic_training_update(X, y, interval_days=7):
    """
    주기적으로 모델을 재학습하는 시스템
    """
    while True:
        # 매주 모델 학습
        model = train_ensemble_models(X, y)
        
        # 모델 성능 기록
        log_performance(epoch, accuracy, success_rate)
        
        # 1주일 동안 대기
        print(f"Waiting for {interval_days} days to update the model.")
        time.sleep(interval_days * 24 * 60 * 60)  # 대기 (1주일)

def dynamic_model_selection_and_optimization(X, y):
    """
    성능을 기준으로 모델을 선택하고, 하이퍼파라미터를 최적화하는 함수
    """
    # 여러 모델 성능 비교 후 최적 모델 선택
    selected_model = select_best_model(models, X, y)
    
    # 최적화된 하이퍼파라미터로 재학습
    optimized_params = optimize_hyperparameters_with_optuna(X, y)
    optimized_model = auto_tune_model(selected_model, X, y)
    
    return optimized_model


def initialize_system(data_path="lotto_data.csv"):
    """
    시스템 초기화: 데이터 로드, 모델 학습 및 예측 준비
    """
    print("Initializing system...")
    
    # 데이터 로드
    lotto_data = load_lotto_data(data_path)
    print(f"Loaded {len(lotto_data)} records from {data_path}.")
    
    # 데이터 전처리
    X, y = prepare_data(lotto_data)
    
    # 모델 학습
    models = train_ensemble_models(X, y)
    print("Initial model training completed.")
    
    return lotto_data, models
 
 
 def weekly_process(models, lotto_data):
    """
    매주 실행되는 자동화 프로세스
    """
    # Step 1: 예측 생성
    X, y = prepare_data(lotto_data)
    recommendations = provide_ensemble_recommendations(models, X)
    print("\n이번 주 추천 번호:")
    for idx, rec in enumerate(recommendations, 1):
        print(f"세트 {idx}: {rec}")
    
    # Step 2: 실제 결과와 비교
    actual_numbers = get_actual_lotto_numbers()  # 실제 로또 번호 입력 (수동 또는 자동 API 연결)
    accuracy, success_rate = evaluate_performance(recommendations, actual_numbers)
    
    # Step 3: 성과 기록
    log_performance(len(lotto_data), accuracy, success_rate)
    
    # Step 4: 사용자 피드백 통합
    user_feedback = get_user_feedback()  # 사용자 피드백 입력
    models = integrate_user_feedback(models, user_feedback, X, y)
    
    # Step 5: 데이터 업데이트
    real_time_data = collect_real_time_data()  # 실시간 데이터 가져오기
    lotto_data = augment_data_with_real_time_data(lotto_data, real_time_data)
    
    # Step 6: 모델 재학습
    models = train_ensemble_models(X, y)
    
    print("Weekly process completed.")
    return models, lotto_data


def monitor_model_performance(models, X_test, y_test):
    """
    모델 성능을 모니터링하고 저하 여부를 확인
    """
    for model in models:
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        print(f"{model.__class__.__name__} RMSE: {rmse:.4f}")
        
        if rmse > 10:  # 성능 기준 설정
            print(f"{model.__class__.__name__} 성능 저하 감지. 하이퍼파라미터 최적화 필요.")
            # 하이퍼파라미터 최적화 실행
            new_params = optimize_hyperparameters(X_test, y_test)
            model = auto_tune_model(model, X_test, y_test)

    return models


from keras.models import Sequential
from keras.layers import LSTM, Dense

def build_lstm_model(input_shape):
    """
    LSTM 기반 딥러닝 모델 설계
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print("LSTM model created.")
    return model


import matplotlib.pyplot as plt

def plot_performance(log_file="performance_log.txt"):
    """
    성능 로그를 시각화
    """
    data = pd.read_csv(log_file, sep="\t")
    plt.plot(data['Epoch'], data['Accuracy'], label="Accuracy")
    plt.plot(data['Epoch'], data['Success Rate'], label="Success Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Performance")
    plt.legend()
    plt.title("Model Performance Over Time")
    plt.show()


import os

def setup_environment():
    """
    작업 환경 초기화
    """
    required_dirs = ["checkpoints", "logs", "data"]
    for d in required_dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")
    print("Environment setup completed.")

# 환경 초기화 실행
setup_environment()


import shutil
from datetime import datetime

def backup_system_state(state_folder="backups"):
    """
    시스템 상태 백업
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(state_folder, f"backup_{timestamp}")
    if not os.path.exists(state_folder):
        os.makedirs(state_folder)
    shutil.copy("model_checkpoint.pkl", backup_path)
    shutil.copy("training_progress.log", backup_path)
    print(f"Backup created at {backup_path}.")


def analyze_prediction_accuracy(predictions, actual_numbers):
    """
    예측 정확도를 계산하고 결과 반환
    """
    correct_numbers = sum([1 for p in predictions if p in actual_numbers])
    accuracy = correct_numbers / len(actual_numbers) * 100
    print(f"Prediction Accuracy: {accuracy:.2f}%")
    return accuracy


import matplotlib.pyplot as plt

def plot_prediction_performance(log_file="performance_log.txt"):
    """
    예측 성과를 시각화
    """
    log_data = pd.read_csv(log_file, sep="\t")
    plt.figure(figsize=(10, 6))
    plt.plot(log_data['Epoch'], log_data['Accuracy'], label="Accuracy", marker='o')
    plt.plot(log_data['Epoch'], log_data['Success Rate'], label="Success Rate", marker='x')
    plt.title("Prediction Performance Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Performance (%)")
    plt.legend()
    plt.grid()
    plt.show()


def evaluate_model_performance(models, X_test, y_test):
    """
    여러 모델의 성능을 평가하고 결과 반환
    """
    performances = {}
    for model in models:
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        performances[model.__class__.__name__] = rmse
        print(f"{model.__class__.__name__} RMSE: {rmse:.4f}")
    return performances


def retrain_models_with_latest_data(models, data):
    """
    최신 데이터를 사용하여 모델 재학습
    """
    X, y = prepare_data(data)
    for model in models:
        model.fit(X, y)
    print("Models retrained with latest data.")
    return models


from keras.models import Sequential
from keras.layers import LSTM, Dense

def build_lstm_model(input_shape):
    """
    LSTM 기반 딥러닝 모델
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print("LSTM model built.")
    return model


import torch

# GPU 활성화 확인
if torch.cuda.is_available():
    print("GPU가 활성화되었습니다:", torch.cuda.get_device_name(0))
else:
    print("GPU를 사용할 수 없습니다.")


:from google.colab import drive

# Google Drive 마운트
drive.mount('/content/drive')

# 작업 디렉토리 설정
import os
project_path = '/content/drive/MyDrive/lotto_project'
if not os.path.exists(project_path):
    os.makedirs(project_path)
print(f"Project directory set to: {project_path}")


import dask.dataframe as dd

# 데이터 로드 및 처리
file_path = '/content/drive/MyDrive/lotto_project/lotto_data.csv'
data = dd.read_csv(file_path)
print(data.head())


# 데이터 샘플링
sampled_data = data.sample(frac=0.2)  # 전체 데이터의 20% 샘플링


from sklearn.ensemble import RandomForestRegressor

# 모델 정의 및 크기 조정
rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)  # 기본값보다 적은 트리와 깊이 사용


import joblib

# 모델 저장
joblib.dump(rf_model, '/content/drive/MyDrive/lotto_project/rf_model.pkl')

# 모델 로드
rf_model = joblib.load('/content/drive/MyDrive/lotto_project/rf_model.pkl')


import xgboost as xgb

# GPU를 사용하는 XGBoost 모델
xgb_model = xgb.XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', n_estimators=100, max_depth=6)
xgb_model.fit(X_train, y_train)


import torch
import torch.nn as nn
import torch.optim as optim

# 간단한 딥러닝 모델 정의
class LottoModel(nn.Module):
    def __init__(self):
        super(LottoModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 출력 크기
        )

    def forward(self, x):
        return self.fc(x)

# 모델 초기화 및 GPU로 이동
model = LottoModel().to('cuda')

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
for epoch in range(100):
    model.train()
    inputs = torch.tensor(X_train.values, dtype=torch.float32).to('cuda')
    targets = torch.tensor(y_train.values, dtype=torch.float32).to('cuda')
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


// 브라우저 콘솔에 입력하여 Colab 세션 유지
function ClickConnect(){
    console.log("코드 실행 중...");
    document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(ClickConnect, 60000);  // 60초마다 연결


# 학습 중단 지점 저장
torch.save(model.state_dict(), '/content/drive/MyDrive/lotto_project/model_checkpoint.pth')

# 모델 복구
model.load_state_dict(torch.load('/content/drive/MyDrive/lotto_project/model_checkpoint.pth'))
model.to('cuda')

batch_size = 64
num_batches = len(X_train) // batch_size

for batch_idx in range(num_batches):
    batch_X = X_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    batch_y = y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size]
    # 배치 단위로 학습

def log_training(epoch, loss, accuracy):
    with open('/content/drive/MyDrive/lotto_project/training_log.txt', 'a') as f:
        f.write(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%\n")


def exclude_explicit_combinations(data, excluded_combinations):
    """
    명시적으로 제외할 조합을 데이터에서 제거
    """
    excluded_df = pd.DataFrame(excluded_combinations, columns=['번호1', '번호2', '번호3', '번호4', '번호5', '번호6'])
    filtered_data = pd.merge(data, excluded_df, on=['번호1', '번호2', '번호3', '번호4', '번호5', '번호6'], how='left', indicator=True)
    filtered_data = filtered_data[filtered_data['_merge'] == 'left_only'].drop(columns=['_merge'])
    print(f"명시적 제외 조합 처리 완료: {len(data) - len(filtered_data)}개 제외")
    return filtered_data


def exclude_dynamic_patterns(data, exclude_continuous=True, exclude_same_group=True):
    """
    동적 패턴 기반으로 조합을 제외
    """
    filtered_data = data.copy()

    if exclude_continuous:
        # 연속 번호 조합 제외 (예: 1, 2, 3, 4, 5, 6)
        continuous = filtered_data.apply(lambda row: all(b - a == 1 for a, b in zip(row[:-1], row[1:])), axis=1)
        filtered_data = filtered_data[~continuous]
        print(f"연속 번호 조합 제외 완료: {continuous.sum()}개 제외")

    if exclude_same_group:
        # 특정 번호 구간만 포함된 조합 제외 (예: 1~10에서만 선택된 조합)
        group_filter = filtered_data.apply(lambda row: max(row) - min(row) <= 10, axis=1)
        filtered_data = filtered_data[~group_filter]
        print(f"특정 구간 조합 제외 완료: {group_filter.sum()}개 제외")

    return filtered_data


import itertools
from collections import Counter

# 과거 당첨 데이터 예시
past_results = [
    [3, 15, 22, 29, 35, 42],
    [7, 11, 23, 35, 38, 42],
    [2, 5, 18, 23, 27, 40],
    # 추가 당첨 데이터...
]

# 조건: 빈도수가 낮은 패턴을 감지하는 함수
def detect_infrequent_patterns(past_results):
    # 끝자리 빈도수 계산
    end_digits = [num % 10 for result in past_results for num in result]
    end_digit_count = Counter(end_digits)
    
    # 연속된 숫자 패턴 계산
    consecutive_patterns = Counter()
    for result in past_results:
        sorted_result = sorted(result)
        for i in range(len(sorted_result) - 2):
            if sorted_result[i+1] == sorted_result[i] + 1 and sorted_result[i+2] == sorted_result[i] + 2:
                consecutive_patterns[tuple(sorted_result[i:i+3])] += 1
                
    return end_digit_count, consecutive_patterns

# 동적으로 필터링 조건 설정
def is_invalid_combination_dynamic(combination, end_digit_count, consecutive_patterns):
    # 끝자리 빈도 기반 필터링
    end_digits = [num % 10 for num in combination]
    if any(end_digits.count(digit) >= 3 and end_digit_count[digit] < 2 for digit in set(end_digits)):
        return True  # 부적합 조합
    
    # 연속된 숫자 패턴 기반 필터링
    sorted_combination = sorted(combination)
    for i in range(len(sorted_combination) - 2):
        if (sorted_combination[i], sorted_combination[i+1], sorted_combination[i+2]) in consecutive_patterns:
            return True
    
    return False

# 전체 조합 생성
all_combinations = list(itertools.combinations(range(1, 46), 6))

# 과거 데이터 기반 패턴 분석
end_digit_count, consecutive_patterns = detect_infrequent_patterns(past_results)

# 부적합 조합 필터링
valid_combinations = [comb for comb in all_combinations if not is_invalid_combination_dynamic(comb, end_digit_count, consecutive_patterns)]

# 결과 출력
print(f"전체 조합 수: {len(all_combinations)}")
print(f"부적합 조합 제외 후 남은 조합 수: {len(valid_combinations)}")


def exclude_low_probability_patterns(data, past_data, threshold=0.01):
    """
    과거 데이터를 기반으로 출현 확률이 낮은 패턴을 감지하여 제외
    :param data: 학습 대상 데이터 (DataFrame)
    :param past_data: 과거 당첨 데이터 (DataFrame)
    :param threshold: 제외 기준 임계값 (출현 확률)
    :return: 제외된 조합이 제거된 DataFrame
    """
    # 1. 패턴 분석: 각 조합의 출현 빈도 계산
    pattern_counts = past_data.value_counts()
    total_combinations = len(past_data)
    pattern_probabilities = pattern_counts / total_combinations
    
    # 2. 임계값 기준으로 제외
    low_probability_patterns = pattern_probabilities[pattern_probabilities < threshold].index
    excluded_data = data[~data.apply(tuple, axis=1).isin(low_probability_patterns)]
    
    print(f"출현 확률 {threshold:.2f} 미만 패턴 제외 완료: {len(data) - len(excluded_data)}개 제외")
    return excluded_data



def update_dynamic_threshold(data, past_data, real_time_data, alpha=0.5):
    """
    새로운 데이터를 기반으로 동적 임계값을 업데이트
    :param data: 현재 데이터 (DataFrame)
    :param past_data: 과거 데이터 (DataFrame)
    :param real_time_data: 실시간 데이터 (DataFrame)
    :param alpha: 가중치 (과거 데이터와 실시간 데이터의 비율)
    :return: 업데이트된 제외 기준
    """
    # 1. 과거 및 실시간 데이터 결합
    combined_data = pd.concat([past_data, real_time_data], ignore_index=True)

    # 2. 새 패턴 확률 계산 (가중치 적용)
    pattern_counts = combined_data.value_counts()
    total_combinations = len(combined_data)
    pattern_probabilities = pattern_counts / total_combinations

    # 3. 기존 데이터와 새로운 데이터의 가중 평균
    past_probabilities = past_data.value_counts() / len(past_data)
    real_time_probabilities = real_time_data.value_counts() / len(real_time_data)
    updated_probabilities = (alpha * past_probabilities + (1 - alpha) * real_time_probabilities)

    print("동적 기준 업데이트 완료")
    return updated_probabilities


def user_filter_options(data, past_data, real_time_data):
    """
    사용자에게 필터링 옵션을 제공
    """
    print("1. 출현 확률이 낮은 패턴 제외")
    print("2. 동적 기준 업데이트")
    print("3. 둘 다 적용")
    choice = int(input("선택: "))

    if choice == 1:
        threshold = float(input("제외 임계값 (예: 0.01): "))
        filtered_data = exclude_low_probability_patterns(data, past_data, threshold)
    elif choice == 2:
        filtered_data = update_dynamic_threshold(data, past_data, real_time_data)
    elif choice == 3:
        threshold = float(input("제외 임계값 (예: 0.01): "))
        filtered_data = exclude_low_probability_patterns(data, past_data, threshold)
        filtered_data = update_dynamic_threshold(filtered_data, past_data, real_time_data)
    else:
        print("잘못된 입력입니다. 원본 데이터 반환.")
        filtered_data = data

    return filtered_data



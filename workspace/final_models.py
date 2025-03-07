import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import yeojohnson
import optuna

train = pd.read_csv('./train.csv').drop(columns=['ID'])
test = pd.read_csv('./test.csv').drop(columns=['ID'])

# 데이터_결측치 처리: 최빈값으로 대체
train = train.fillna(train.mode().iloc[0])
test = test.fillna(test.mode().iloc[0])

X = train.drop('임신 성공 여부', axis=1)
y = train['임신 성공 여부']

# 카테고리형 컬럼 
categorical_columns = [
    "시술 시기 코드", "시술 당시 나이", "시술 유형", "특정 시술 유형", "배란 자극 여부", "배란 유도 유형", "단일 배아 이식 여부", "착상 전 유전 검사 사용 여부",
    "착상 전 유전 진단 사용 여부", "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", "여성 부 불임 원인", "부부 주 불임 원인", "부부 부 불임 원인", "불명확 불임 원인",
    "불임 원인 - 난관 질환", "불임 원인 - 남성 요인", "불임 원인 - 배란 장애", "불임 원인 - 여성 요인", "불임 원인 - 자궁경부 문제", "불임 원인 - 자궁내막증", "불임 원인 - 정자 농도",
    "불임 원인 - 정자 면역학적 요인", "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태", "배아 생성 주요 이유", "총 시술 횟수", "클리닉 내 총 시술 횟수", "IVF 시술 횟수", "DI 시술 횟수",
    "총 임신 횟수", "IVF 임신 횟수", "DI 임신 횟수", "총 출산 횟수", "IVF 출산 횟수", "DI 출산 횟수", "난자 출처", "정자 출처", "난자 기증자 나이", "정자 기증자 나이",
    "동결 배아 사용 여부", "신선 배아 사용 여부", "기증 배아 사용 여부", "대리모 여부", "PGD 시술 여부", "PGS 시술 여부"]

for col in categorical_columns:
    X[col] = X[col].astype(str)
    test[col] = test[col].astype(str)

# 카테고리형 컬럼 인코딩
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train_encoded = X.copy()
X_train_encoded[categorical_columns] = ordinal_encoder.fit_transform(X[categorical_columns])
X_test_encoded = test.copy()
X_test_encoded[categorical_columns] = ordinal_encoder.transform(test[categorical_columns])

# 수치형 컬럼 
numeric_columns = ["임신 시도 또는 마지막 임신 경과 연수", "총 생성 배아 수", "미세주입된 난자 수",
                   "미세주입에서 생성된 배아 수", "이식된 배아 수", "미세주입 배아 이식 수",
                   "저장된 배아 수", "미세주입 후 저장된 배아 수", "해동된 배아 수", "해동 난자 수",
                   "수집된 신선 난자 수",  "저장된 신선 난자 수", "혼합된 난자 수", "파트너 정자와 혼합된 난자 수",
                   "기증자 정자와 혼합된 난자 수", "난자 채취 경과일", "난자 해동 경과일", "난자 혼합 경과일",
                   "배아 이식 경과일", "배아 해동 경과일"]

# 다항식 변환 추가
poly = PolynomialFeatures(degree=2, include_bias=False)

# 수치형 변수에 대해 다항식 특성 생성
X_poly = poly.fit_transform(X[numeric_columns])
test_poly = poly.transform(test[numeric_columns])

# 다항식 특성을 기존 데이터와 결합
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(numeric_columns))
test_poly_df = pd.DataFrame(test_poly, columns=poly.get_feature_names_out(numeric_columns))

# 기존 수치형 특성과 결합
X = pd.concat([X, X_poly_df], axis=1)
test = pd.concat([test, test_poly_df], axis=1)

# 스케일링
scaler = StandardScaler()
X[numeric_columns] = scaler.fit_transform(X[numeric_columns])
test[numeric_columns] = scaler.fit_transform(test[numeric_columns])

# K-Fold 교차 검증
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(X_train_encoded.shape[0])  # OOF 예측을 위한 배열
test_preds = np.zeros(X_test_encoded.shape[0])  # 최종 예측을 위한 배열

# Optuna objective function (모델 최적화)
def objective(trial):
    # ExtraTreesClassifier 하이퍼파라미터 최적화
    extra_tree_model = ExtraTreesClassifier(
        n_estimators=trial.suggest_int('extra_tree_n_estimators', 50, 500),
        max_depth=trial.suggest_int('extra_tree_max_depth', 3, 15),
        min_samples_split=trial.suggest_int('extra_tree_min_samples_split', 2, 20),
        min_samples_leaf=trial.suggest_int('extra_tree_min_samples_leaf', 1, 20),
        random_state=42
    )

    # CatBoostClassifier 하이퍼파라미터 최적화
    cat_model = CatBoostClassifier(
        iterations=1000,
        depth=trial.suggest_int('cat_depth', 3, 12),
        learning_rate=trial.suggest_loguniform('cat_learning_rate', 1e-3, 0.1),
        l2_leaf_reg=trial.suggest_int('cat_l2_leaf_reg', 1, 100),
        rsm=trial.suggest_uniform('cat_rsm', 0.5, 1.0),
        random_seed=42,
        verbose=0
    )

    # 가중치 최적화
    weight_extra_tree = trial.suggest_float('weight_extra_tree', 0.0, 1.0)
    weight_cat = 1 - weight_extra_tree

    # KFold 교차 검증
    auc_scores = []
    oof_preds_fold = np.zeros(X_train_encoded.shape[0])
    for train_index, val_index in kf.split(X_train_encoded, y):
        X_train_fold, X_val_fold = X_train_encoded.iloc[train_index], X_train_encoded.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        # 모델 학습
        extra_tree_model.fit(X_train_fold, y_train_fold)
        cat_model.fit(X_train_fold, y_train_fold)

        # 예측
        extra_tree_val_preds = extra_tree_model.predict_proba(X_val_fold)[:, 1]
        cat_val_preds = cat_model.predict_proba(X_val_fold)[:, 1]

        # 가중 평균으로 예측값 결합 (Blending)
        blended_val_preds = (weight_extra_tree * extra_tree_val_preds) + (weight_cat * cat_val_preds)
        
        oof_preds_fold[val_index] = blended_val_preds
        auc_scores.append(roc_auc_score(y_val_fold, blended_val_preds))

    # 평균 AUC 반환
    return np.mean(auc_scores)

# Optuna study 시작
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # 최적화를 50번 반복

# 최적의 하이퍼파라미터로 모델 재학습
best_params = study.best_params
print(f"Best parameters: {best_params}")

# 최적의 하이퍼파라미터로 모델 학습 및 예측
final_extra_tree_model = ExtraTreesClassifier(
    n_estimators=best_params['extra_tree_n_estimators'],
    max_depth=best_params['extra_tree_max_depth'],
    min_samples_split=best_params['extra_tree_min_samples_split'],
    min_samples_leaf=best_params['extra_tree_min_samples_leaf'],
    random_state=42
)

final_cat_model = CatBoostClassifier(
    iterations=1000,
    depth=best_params['cat_depth'],
    learning_rate=best_params['cat_learning_rate'],
    l2_leaf_reg=best_params['cat_l2_leaf_reg'],
    random_seed=42,
    verbose=0
)

# 전체 훈련 데이터로 모델 학습
final_extra_tree_model.fit(X_train_encoded, y)
final_cat_model.fit(X_train_encoded, y)

# 테스트 데이터에 대해 예측
final_extra_tree_preds = final_extra_tree_model.predict_proba(X_test_encoded)[:, 1]
final_cat_preds = final_cat_model.predict_proba(X_test_encoded)[:, 1]

# 최적화된 가중치를 사용해 최종 예측값 결합
final_weight_extra_tree = best_params['weight_extra_tree']
final_weight_cat = 1 - final_weight_extra_tree
final_preds = (final_weight_extra_tree * final_extra_tree_preds) + (final_weight_cat * final_cat_preds)

# Save or output the predictions
test_preds = final_preds

# 모델 성능 평가
roc_auc = roc_auc_score(y, oof_preds)
print(f"ROC AUC Score: {roc_auc}")

# 최종 예측 결과 (test.csv에 대한 예측)
final_predictions = test_preds

# 'ID' 컬럼을 다시 불러오기 (테스트 데이터에서 가져옴)
test_id = pd.read_csv('./test.csv')['ID']

# 확률값을 CSV로 저장
submission = pd.DataFrame({'ID': test_id, 'probability': final_predictions})
submission.to_csv('./submission_0225_3.csv', index=False)
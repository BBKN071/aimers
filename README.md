![header](https://capsule-render.vercel.app/api?type=rounded&height=300&color=gradient&text=AIMERS)

# aimers
LG aimers 6기 코드 자료

### 테스트 기록
1 : 결측치 존재 항목 전부 제거 -> baseline 미만   
2 : 결측치 20만개 이상 항목 제거, 나머지 결측치는 평균으로 처리 -> 0.688 ~ baseline   
3 : 기본 데이터 + 모델 4개 앙상블 -> 0.729   
4, 5 : 2번 데이터 + lightgbm regressor -> 0.739   
6 : 2번 데이터 + Xgboost 단일 + 튜닝 -> 0.736   
7 : 2번 데이터 + LightGBM classifier + optuna -> 0.7403   
8 : 2번 데이터 + catboost + optuna -> 0.7403   
9 : 데이터 결측치 최빈값 + catboost+optuna -> 0.74109    
10 : 9번 데이터 + lightgbm -> 0.7404    
11 : 9번 데이터 + gradient boosting -> 0.7397   
12, 13, 14, 15 : 9번 데이터 + catboost 파라미터 최적화 -> 0.740, 0.739, 0.740, 0.7405   
16 : 9번 데이터 + lightgbm regressor -> 0.7405   
17 : 9번 데이터 + catboost, ROC curve 기반 최적화 -> 0.739   

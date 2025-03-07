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
18, 19 : 숫자형 변수 정규화 + catboost -> 0.7404, 0.7408    
20 : 숫자형 변수 yeo-johnson 변환 -> 0.7410   
21, 22 : 데이터 필요없는 컬럼 삭제, 중요 컬럼 중복 -> 0.72~   
23, 24 : yeo-johnson, quantile transform -> 0.68~(데이터 오류?)   
25 : standard scaler + K-fold(5) -> 0.739   
26 : 9번 데이터 + K-Fold(5) -> 0.740   
27 ~ 29 : 각 케이스 별 K-Fold(5) 시도 <= 0.740   
30 : 9번 데이터 동일 모델 20번 학습 후 앙상블 -> 0.7409(일부 향상)    
31 : 50개 앙상블 -> 0.7409(30번 대비 소폭 하락, 앙상블 개수 유의)   

32 : 20번대 학습 모델 기반 중요도 0.1 이하 데이터 컬럼 삭제(카테고리 컬럼 47 -> 25) -> 0.7411 (9번보다 성능 향상)   
33 : 32번 데이터 20개 모델 앙상블 -> 0.7410    
34 : 32번 모델 기준 중요도 0,1 이하 데이터 컬럼 삭제(카테고리 컬럼 47 -> 16) -> 0.7403   
35 : 34번 데이터 20개 모델 앙상블 -> 0.7401   
36 : 32번 데이터 + standard scaler 적용 -> 0.739   
37 : 데이터 수정 -> 0.7403   
38 : 37번 데이터 + 중요도 0인 피쳐 11개 삭제 -> 0.7403   
39-41 : 37번 데이터 + catboost + a 앙상블 -> < 0.741   
   
## 최종   
데이터 : 컬럼 유지 + 다항식 결합 + 표준 스케일링   
모델 :   
1. Extratrees classifier + optuna(하이퍼 파라미터 튜닝)   
2. catboost classifier + optuna(하이퍼 파라미터 튜닝)   
1 + 2 모델의 가중 앙상블(weighted ensemble) 기법 -> 앙상블 모델 가중치들의 합이 1이 되도록 설정   
   
### 결과 : AUC 0.7415 (193 / 1570) 


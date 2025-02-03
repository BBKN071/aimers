![header](https://capsule-render.vercel.app/api?type=rounded&height=300&color=gradient&text=AIMERS)

# aimers
LG aimers 6기 코드 자료

### 테스트 기록
1 : 결측치 존재 항목 전부 제거 -> baseline 미만   
2 : 결측치 20만개 이상 항목 제거, 나머지 결측치는 평균으로 처리 -> 0.688 ~ baseline   
3 : 기본 데이터 + 모델 4개 앙상블 -> 0.729   
4 : 2번 데이터 + lightgbm regressor -> 0.739   


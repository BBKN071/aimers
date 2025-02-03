# 베이스 이미지 선택 (Jupyter + 머신러닝 패키지가 포함된 Python 환경)
FROM python:3.10

# 작업 디렉토리 설정
WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888

# 컨테이너 실행 시 Jupyter Notebook 시작
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=''"]
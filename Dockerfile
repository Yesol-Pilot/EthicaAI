# EthicaAI Docker 기반 재현성 패키지
# Python 3.10 + JAX CPU + 전체 분석 파이프라인

FROM python:3.10-slim

LABEL maintainer="dpfh1537@gmail.com"
LABEL description="EthicaAI: Computational Verification of Sen's Meta-Ranking Theory"
LABEL version="1.0"

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /ethicaai

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 코드 복사
COPY . .

# 환경 변수
ENV PYTHONIOENCODING=utf-8
ENV JAX_PLATFORM_NAME=cpu
ENV PYTHONPATH=/ethicaai

# 기본 실행: 전체 분석 파이프라인
CMD ["python", "reproduce.py"]

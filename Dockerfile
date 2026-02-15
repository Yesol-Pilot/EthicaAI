# EthicaAI Docker 재현성 패키지
# NeurIPS 2026 — 38 모듈, 74 Figure 완전 재현
# 빌드: docker build -t ethicaai .
# 실행: docker run -v $(pwd)/output:/ethicaai/simulation/outputs/reproduce ethicaai

FROM python:3.10-slim

LABEL maintainer="dpfh1537@gmail.com"
LABEL description="EthicaAI: Computational Verification of Sen's Meta-Ranking Theory"
LABEL version="2.0"
LABEL paper="NeurIPS 2026"

# 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /ethicaai

# Python 의존성 설치 (캐시 최적화를 위해 먼저 복사)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 코드 복사
COPY . .

# 출력 디렉토리 보장
RUN mkdir -p simulation/outputs/reproduce site/figures

# 환경 변수
ENV PYTHONIOENCODING=utf-8
ENV JAX_PLATFORM_NAME=cpu
ENV PYTHONPATH=/ethicaai
ENV ETHICAAI_OUTPUT_DIR=/ethicaai/simulation/outputs/reproduce

# 헬스체크: Python + JAX import 가능 확인
HEALTHCHECK --interval=30s --timeout=10s \
    CMD python -c "import jax; print(f'JAX {jax.__version__} OK')" || exit 1

# 기본 실행: 전체 분석 파이프라인
CMD ["python", "reproduce.py"]

# EthicaAI 워크스페이스 룰

## 프로젝트 정체성
- **프로젝트명**: EthicaAI — 최선합리성 이론의 계산적 검증
- **한 줄**: 멀티에이전트 RL 시뮬레이션으로 "윤리적 에이전트가 이기적 에이전트를 장기적으로 이긴다"를 GPU 학습으로 증명
- **형태**: 학술 논문 + 시뮬레이션 코드 + 시각화 대시보드
- **자매 프로젝트**: WhyLab (인과추론) — 시뮬레이션 결과를 DML로 분석

## 필수 규칙

### 1. 언어
- 모든 코드 주석, 커밋, 문서: **한국어**
- 수학 용어/변수명: 영문 유지 (Nash Equilibrium, CATE, Reward)
- 논문 본문: **한국어** (영문 번역본 별도)

### 2. 프로젝트 구조
- `paper/`: 논문 원고 + 수식 정의 + 참고 문헌
- `simulation/`: 멀티에이전트 RL 시뮬레이션 코드
- `analysis/`: Jupyter 노트북 (결과 분석 + 인과추론)
- `dashboard/`: 시각화 웹 (선택사항)
- `original/`: 기존 에세이 원문 보관

### 3. Python 코드
- Python 3.11, PyTorch CUDA 12.1
- **모든 함수에 타입 힌트 필수**
- 독스트링: Google 스타일
- 에이전트 프레임워크: PettingZoo 또는 OpenSpiel
- RL 프레임워크: Stable-Baselines3 또는 RLlib

### 4. 시뮬레이션 설계 원칙
- 에이전트 유형 최소 3종: 이기적(Selfish), 최선합리(Optimal), 제한합리(Bounded)
- 환경 최소 3종: 반복 죄수의 딜레마, 공공재 게임, 신뢰 게임
- 라운드 수: 최소 10만 이상 (통계적 유의성 확보)
- 모든 실험은 **시드(seed) 고정** + 재현 가능

### 5. 수학적 엄밀함
- 최선합리성 효용 함수: U*(a) = α·U_self + β·U_social + γ·U_ethical - δ·Risk_longterm
- 모든 결과에 95% 신뢰구간 포함
- WhyLab의 DML로 인과 효과 검증 연계

### 6. Git
- 커밋: `<타입>(<범위>): <한국어 설명>`
- 범위: paper, simulation, analysis, dashboard

### 7. GPU
- RTX 4070 SUPER (12GB), CUDA 12.1, FP16

### 8. 보안
- API 키 커밋 금지, 합성/시뮬레이션 데이터만

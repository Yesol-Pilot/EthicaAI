# EthicaAI — Twitter/X 홍보 쓰레드 초안

> 실험 결과 확정 후 수치 업데이트 필요. 아래는 초안.

---

## Thread (영어)

### 1/6 🧵
Can AI agents learn *morality* instead of just maximizing rewards?

We computationally verified Amartya Sen's "Meta-Ranking" theory—
preferences over preferences—in a 100-agent social dilemma.

3 surprising findings ↓

### 2/6 — Finding 1: Dynamic > Static
Injecting fixed social values (SVO) into agents does nothing meaningful (p=0.64).

But giving agents the ability to *dynamically switch* between self-interest and morality?

That changes everything. (p=0.0023, HAC Robust SE)

### 3/6 — Finding 2: Role Specialization
We expected prosocial agents to cooperate more.

Instead, they *specialize*: some become cleaners 🧹, others become harvesters 🌾

This emergent division of labor is structurally more efficient than uniform cooperation.

[Fig 9 — Role Specialization 첨부]

### 4/6 — Finding 3: Only "Situational Commitment" Survives
Full altruists? They go extinct. Purely selfish? Also die out.

The only Evolutionarily Stable Strategy:
"Help when I can afford it, prioritize survival when I can't."

Rational morality > absolute morality.

### 5/6 — Why It Matters for AI Alignment
Current approaches to AI alignment either:
- Hardcode rules (brittle) 📏
- Learn from human feedback (costly) 🏷️

Meta-Ranking offers a middle path:
Let agents *learn* when to be moral—not just *what* morality is.

### 6/6 — Links
📄 Paper (Zenodo): [DOI 링크]
💻 Code: https://github.com/Yesol-Pilot/EthicaAI
🧪 JAX + PPO, reproducible in ~35 min on RTX 4070

Looking for cs.MA endorser for arXiv! DM if interested 🙏

---

## 해시태그
#MARL #AIAlignment #GameTheory #ReinforcementLearning #AmartyaSen
#MultiAgentSystems #AIEthics #NeurIPS2026

---

## 한국어 (선택)

### 1/4
AI 에이전트가 보상 최대화가 아니라 '도덕'을 배울 수 있을까?

노벨경제학상 수상자 아마르티아 센의 "메타랭킹" 이론을 
100-에이전트 사회적 딜레마에서 계산적으로 검증했습니다.

### 2/4
핵심 발견:
1️⃣ 고정된 사회적 가치 주입은 효과 없음
2️⃣ 동적 도덕 전환(λ_t)만이 집단 복지를 유의하게 향상
3️⃣ "상황적 헌신"만이 진화적으로 안정 — 절대적 이타주의는 멸종

### 3/4
AI 정렬(alignment)에 대한 시사점:
규칙을 하드코딩하지 마세요. 
에이전트가 "언제 도덕적이어야 하는지"를 학습하게 하세요.

### 4/4
📄 논문: Zenodo (DOI)
💻 코드: GitHub
독립 연구자로서 첫 논문입니다. 피드백 환영합니다!

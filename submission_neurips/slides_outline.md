# EthicaAI — NeurIPS 2026 Workshop 발표 아웃라인

> 5분 Spotlight 또는 포스터 발표용

## Slide 1: Title + Hook (30초)
**"Can AI agents learn WHEN to be moral?"**
- 제목 + 저자 + 소속
- 핵심 질문: "도덕에 대한 결정을 에이전트에게 맡기면 어떤 일이 생길까?"

## Slide 2: Problem (45초)
- Tragedy of the Commons 시각화
- 기존 접근법의 한계: 규칙 주입(brittle) vs RLHF(costly)
- Sen의 Meta-Ranking 직관: "선호에 대한 선호"

## Slide 3: Method (60초)
- 보상 함수 수식: R_total = (1-λ_t)·U_self + λ_t·[U_meta - ψ]
- 동적 λ_t: 생존 위협 → 이기적 / 여유 → 이타적
- 실험 설계: 7 SVO × 10 seeds × {20, 100} agents

## Slide 4: Key Result 1 — Dynamic > Static (45초)
- Fig: ATE Forest Plot
- "동적 메타랭킹만이 유의한 효과" (p=0.0023 vs 0.64)
- 3중 통계 검증 (OLS/LMM/Bootstrap)

## Slide 5: Key Result 2 — Role Specialization (45초)
- Fig 9: Cleaner vs Harvester 분화
- "균일 협력이 아닌 역할 분화가 최적"
- 20-agent와 100-agent 에서 일관된 패턴

## Slide 6: Key Result 3 — ESS (45초)
- "절대적 이타주의는 멸종한다"
- 상황적 헌신(Situational Commitment)만 생존
- AI Alignment 시사점: corrigibility와의 연결

## Slide 7: Conclusion + Q&A (30초)
- 3가지 Takeaway 요약
- Code & Paper 링크 (QR 코드)
- "Questions?"

---

## 포스터 구성 (대안)

| 패널 | 내용 |
|------|------|
| 좌상 | Title + Abstract + Contribution |
| 좌하 | Method 수식 + 환경 시각화 |
| 중앙 | Fig 10 (Scale Comparison) — 메인 결과 |
| 우상 | Fig 9 (Role Specialization) |
| 우하 | ATE Forest Plot + 통계 Table |
| 하단 | Conclusion + GitHub/Zenodo/QR |

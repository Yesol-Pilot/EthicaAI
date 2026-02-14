# EthicaAI: When Should AI Agents Be Moral? 🧠⚖️

[![NeurIPS 2026](https://img.shields.io/badge/Target-NeurIPS_2026-blue?style=for-the-badge&logo=neurips)](https://neurips.cc)
[![30 Figures](https://img.shields.io/badge/Figures-30-brightgreen?style=for-the-badge)](https://ethicaai.vercel.app)
[![560+ Experiments](https://img.shields.io/badge/Experiments-560+-orange?style=for-the-badge)]()
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![JAX](https://img.shields.io/badge/JAX-Accelerated-9cf?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> **"The question isn't *whether* AI should be moral, but *when*."**

**EthicaAI** formalizes Amartya Sen's **Meta-Ranking** theory (preferences over preferences) as a dynamic mechanism in Multi-Agent Reinforcement Learning. We demonstrate that _Situational Commitment_ — morality conditional on survival — is the only Evolutionarily Stable Strategy across 4 environments, 7 SVO conditions, and up to 1,000 agents.

<p align="center">
  <a href="https://ethicaai.vercel.app"><strong>🌐 Interactive Dashboard (30 Figures)</strong></a> &nbsp;|&nbsp;
  <a href="paper_english.md"><strong>📄 Full Paper</strong></a> &nbsp;|&nbsp;
  <a href="submission_neurips/main.tex"><strong>📝 LaTeX</strong></a>
</p>

---

## 🔬 Five Key Findings

| # | Finding | Evidence |
|:-:|---------|---------|
| **1** | Dynamic meta-ranking (λ_t) significantly enhances collective welfare | p=0.0003, Cohen's f²=0.40 |
| **2** | Agents exhibit emergent **role specialization** (Cleaners vs Eaters) | σ divergence p<0.0001 |
| **3** | Only "Situational Commitment" survives as **ESS** (~12% of population) | 200-gen replicator dynamics |
| **4** | Individualist SVO (θ=15°) best matches **human PGG data** | WD=0.053 |
| **5** | SVO rotation accounts for **86%** of total effect | Full factorial 2³ decomposition |

---

## 🌟 Extended Results (Phase M)

### Full Environmental Sweep (560 runs)
4 environments × 7 SVO × 10 seeds — the most comprehensive test of meta-ranking to date.

| Environment | Best ATE (Cooperation) | Optimal SVO | ATE (Reward) |
|:-:|:-:|:-:|:-:|
| Cleanup | +0.083 | Cooperative (60°) | — |
| **PGG** | **+0.211** | **Prosocial (45°)** | **+2.535** |
| **Harvest** | **+0.506** | **Selfish (0°)** | **+0.101** |

### Mixed-SVO Populations: Tipping Point
A **nonlinear tipping point** at ~30% prosocial fraction triggers population-wide cooperation. PGG welfare improvement: **ΔW = +10,080**.

### Communication Channels
1-bit cheap talk boosts cooperation by **+5.8%** for prosocial agents. Message truthfulness converges to **98%** — honesty is evolutionarily favored.

### Continuous Action Spaces
Beta-distribution policies in continuous PGG maintain meta-ranking's ATE ≈ **+0.20**, confirming generalization beyond discrete decisions.

---

## 🛠️ Installation

```bash
# Clone & setup
git clone https://github.com/Yesol-Pilot/EthicaAI.git
cd EthicaAI

python -m venv ethica_env
source ethica_env/bin/activate  # Windows: ethica_env\Scripts\activate
pip install -r requirements.txt
```

**Requirements**: Python 3.10+, CUDA 12+ (optional, for GPU acceleration)

---

## 🚀 Quick Start

### Reproduce All Results (One Command)
```bash
# All 11 analysis modules (Phase G + H + M)
python reproduce.py

# Phase M only (4 new experiments)
python reproduce.py --phase M

# Quick demo
python reproduce.py --quick
```

### Run Individual Experiments
```bash
# M1: Full Sweep (4 envs × 7 SVO × 10 seeds)
python -m simulation.jax.analysis.run_full_sweep simulation/outputs/reproduce

# M2: Mixed-SVO tipping point analysis
python -m simulation.jax.analysis.mixed_svo_experiment simulation/outputs/reproduce

# M3: Communication channels (cheap talk)
python -m simulation.jax.analysis.communication_experiment simulation/outputs/reproduce

# M4: Continuous PGG (Beta-distribution policies)
python -m simulation.jax.analysis.continuous_experiment simulation/outputs/reproduce
```

### Full Training Pipeline (100 Agents)
```bash
python -m simulation.jax.run_full_pipeline large_full      # Meta-Ranking ON
python -m simulation.jax.run_full_pipeline large_baseline   # Baseline (OFF)
```

### Prepare arXiv Submission
```bash
python prepare_arxiv.py  # Generates ethicaai_arxiv.tar.gz
```

---

## 📊 30 Figures

All figures are interactive at [ethicaai.vercel.app](https://ethicaai.vercel.app).

| Phase | Figures | Content |
|:-----:|:-------:|---------|
| **Core** (A–D) | Fig 1–9 | Learning curves, cooperation rates, role specialization, Gini, causal forest |
| **Scale** (E) | Fig 10–11 | 100-agent scale comparison, ATE analysis |
| **Robustness** (G) | Fig 12–16 | Convergence, static/dynamic λ, sensitivity, cross-environment |
| **Extended** (H) | Fig 17–23 | PGG, evolution, mechanism decomposition, Harvest, Melting Pot, Constitutional AI |
| **Deep** (M) | Fig 24–30 | Full sweep heatmap, mixed-SVO tipping point, communication, continuous PGG |

---

## 📂 Repository Structure

```
EthicaAI/
├── simulation/
│   └── jax/
│       ├── analysis/              # 11 analysis modules
│       │   ├── run_full_sweep.py         # M1: Full environmental sweep
│       │   ├── mixed_svo_experiment.py   # M2: Mixed-SVO populations
│       │   ├── communication_experiment.py # M3: Cheap talk
│       │   ├── continuous_experiment.py  # M4: Continuous PGG
│       │   ├── convergence_proof.py      # Convergence verification
│       │   ├── sensitivity_analysis.py   # Parameter sensitivity
│       │   └── ...                       # 5 more modules
│       ├── environments/          # Cleanup, IPD, PGG, Harvest
│       ├── training/              # MAPPO training pipeline
│       └── run_full_pipeline.py   # End-to-end execution
├── submission_neurips/            # LaTeX (NeurIPS 2026 format)
├── submission_arxiv/              # arXiv package (32 figures)
├── site/                          # Interactive dashboard (Vercel)
├── reproduce.py                   # One-command reproduction (11 modules)
├── prepare_arxiv.py               # arXiv package generator
├── paper_english.md               # Full paper (English)
├── paper_korean.md                # Full paper (Korean)
└── twitter_thread_draft.md        # Social media draft
```

---

## 📈 Reproduction Pipeline

```
$ python reproduce.py --phase M
============================================================
  EthicaAI Reproduction Pipeline
  Phase: M  |  Mode: Full
============================================================
  ✓ M1: Full Sweep (4환경 × 7SVO × 10seeds)     — 15.2s
  ✓ M2: Mixed-SVO Population (임계점 분석)        — 10.1s
  ✓ M3: Communication Channels (Cheap Talk)      —  8.3s
  ✓ M4: Continuous PGG (연속 행동 공간)           —  6.8s

  Total: 4/4 succeeded (45.4s)
  🎉 전체 재현 성공!
```

---

## 📜 Citation

```bibtex
@article{heo2026ethicaai,
  title={Beyond Homo Economicus: Computational Verification of Amartya Sen's
         Meta-Ranking Theory via Multi-Agent Reinforcement Learning},
  author={Heo, Yesol},
  journal={arXiv preprint arXiv:2602.XXXXX},
  year={2026},
  note={30 figures, 560+ experiments, 4 environments, 11 reproduction modules}
}
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

> *Built with ❤️ by the Antigravity Team.*

# EthicaAI: Emergent Morality via Meta-Ranking 🧠⚖️

[![NeurIPS 2026 Prep](https://img.shields.io/badge/Status-NeurIPS__2026__Prep-blue?style=for-the-badge&logo=neurips)](https://neurips.cc)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![JAX Powered](https://img.shields.io/badge/JAX-Accelerated-9cf?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)

> **"Reason is not just a slave of the passions, but a sovereign that can choose between them."** — *Amartya Sen*

**EthicaAI** is the official implementation of the paper **"Computational Verification of Amartya Sen's Optimal Rationality via Multi-Agent Reinforcement Learning with Meta-Ranking."**

This project bridges **Moral Philosophy** and **Multi-Agent Reinforcement Learning (MARL)**. By formalizing Amartya Sen's theory of **Meta-Ranking** (preferences over preferences), we demonstrate how AI agents can evolve distinct moral commitments ("Situational Commitment") to solve the **Tragedy of the Commons** in large-scale social dilemmas.

<p align="center">
  <img src="simulation/outputs/run_large_1771042566/figures/fig10_scale_comparison.png" width="800" alt="Scale Comparison">
  <br>
  <em>Fig: Meta-Ranking prevents the "Tragedy of the Commons" at scale (100 Agents). High SVO agents with meta-ranking (blue) sustain resources, while naive agents (gray) collapse.</em>
</p>

---

## 🌟 Key Innovations

### 1. 🧠 **Meta-Ranking Architecture**
Unlike traditional methods that treat morality as a fixed parameter (Static SVO), EthicaAI implements a **dynamic $\lambda_t$ mechanism** that modulates the weight between self-interest and social welfare based on resource abundance.
*   **Survival Mode**: Prioritize self-preservation ($w < w_{survival}$)
*   **Abundance Mode**: Activate moral commitment ($w > w_{boost}$)

### 2. 📈 **Scalability Verified (100 Agents)**
We scaled the simulation from 20 to **100 agents**, confirming that the emergence of cooperation is robust.
*   **Super-Linear Inequality Reduction**: The mechanism becomes *more* effective at maintaining fairness as society grows ($f^2$: 5.79 $\to$ 10.2).
*   **Role Specialization**: Emergence of distinct "Cleaner" and "Eater" classes ($p < 0.0001$).

### 3. 🤝 **Human-AI Alignment**
We validated our agents against **Human Public Goods Game (PGG)** data (Zenodo Dataset, 2025).
*   **Wasserstein Distance < 0.2**: Our agents' "Situational Commitment" mirrors human "Conditional Cooperation."

### 4. 📊 **Rigorous Causal Inference**
We moved beyond simple correlation.
*   **HAC Robust Standard Errors**: Correcting for temporal autocorrelation.
*   **Linear Mixed-Effects Models (LMM)**: Accounting for agent-specific random effects.
*   **Bootstrap Confidence Intervals**: Ensuring statistical solidity.

---

## 🛠️ Installation

Prerequisites: **Python 3.10+**, **CUDA 12+** (for GPU acceleration).

```bash
# 1. Clone the repository
git clone https://github.com/Yesol-Pilot/EthicaAI.git
cd EthicaAI

# 2. Create a virtual environment
python -m venv ethica_env
source ethica_env/bin/activate  # Windows: ethica_env\Scripts\activate

# 3. Install dependencies (JAX, Flax, Statsmodels, etc.)
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run Full Experiment (100 Agents)
Execute the full pipeline including training, evaluation, Causal ATE analysis, and figure generation.

```bash
# Run large-scale experiment (Meta-Ranking ON)
python -m simulation.jax.run_full_pipeline large_full

# Run baseline comparison (Meta-Ranking OFF)
python -m simulation.jax.run_full_pipeline large_baseline
```

### 2. Run Human-AI Comparison
Verify the alignment between simulation results and human data.

```bash
python -m simulation.jax.analysis.human_ai_comparison data/human_pgg.csv simulation/outputs/latest_run/sweep.json
```

### 3. Re-generate Figures (Publication Ready)
Generate NeurIPS-style figures (Times New Roman, 300 DPI, PDF/PNG).

```bash
python -m simulation.jax.analysis.paper_figures simulation/outputs/latest_run
```

---

## 📂 Repository Structure

```
EthicaAI/
├── simulation/
│   ├── jax/                # Core MAPPO Algorithm & Environment (JAX)
│   │   ├── analysis/       # Statistical Analysis (LMM, Bootstrap, Causal)
│   │   ├── config.py       # Experiment Hyperparameters
│   │   └── run_full_pipeline.py # End-to-End Execution Script
│   └── llm/                # (Experimental) Constitutional AI Prototype
├── submission_neurips/     # LaTeX Sources for NeurIPS 2026
├── figures/                # Generated Figures for Paper
└── requirements.txt        # Python Dependencies
```

---

## 📜 Citation

If you use this code or findings, please cite:

```bibtex
@article{heo2026ethicaai,
  title={Computational Verification of Amartya Sen's Optimal Rationality via Multi-Agent Reinforcement Learning with Meta-Ranking},
  author={Heo, Yesol},
  journal={arXiv preprint arXiv:2602.XXXXX},
  year={2026},
  note={Prepared for NeurIPS 2026 Workshop}
}
```

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

> *Built with ❤️ by the Antigravity Team.*

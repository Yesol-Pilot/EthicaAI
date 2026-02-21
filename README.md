# 🧬 EthicaAI: The Genesis Lab
> *"Where AI Agents Evolve Their Own Ethics"*

![Status](https://img.shields.io/badge/Status-Autnomous_Evolution-success?style=for-the-badge&logo=prometheus)
![Language](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Engine](https://img.shields.io/badge/Engine-JAX_GPU-red?style=for-the-badge&logo=nvidia)
![Brain](https://img.shields.io/badge/Brain-Gemini_2.0-orange?style=for-the-badge&logo=google-gemini)

## 🏛️ What is Genesis?

**EthicaAI Genesis** is an autonomous research laboratory where AI agents live, interact, and evolve social contracts without human intervention.
Governed by a hyper-intelligent **Theorist (LLM)**, the system automatically formulates hypotheses, runs massive GPU simulations, and pivots strategies to solve the "Cooperation Dilemma".

### 🧠 The Autonomous Loop
1.  **Thinking (Theorist)**: Gemini 2.0 analyzes history and proposes a new social structure (e.g., "Let's try Inequity Aversion!").
2.  **Simulation (Engineer)**: JAX-accelerated engine runs 20 simultaneous societies (20,000+ steps) in seconds.
3.  **Judgment (Critic)**: Evaluates stability, Gini coefficient, and welfare.
4.  **Intervention (Coordinator)**: Pokes, shocks, or resets the world if stagnation is detected.

---

## 🚀 Key Features (v2.0)

| Feature | Description | Status |
|:---|:---|:---:|
| **GPU Revolution** | **100x Faster** simulations using JAX on RTX 4070 SUPER. | ✅ |
| **Self-Correction** | Automatically switches between *Adaptive*, *Inverse*, and *Institutional* modes. | ✅ |
| **Inequity Aversion** | Agents feel *Envy* and *Guilt*, driving spontaneous fairness. | ✅ |
| **Live Dashboard** | Real-time visualization of the evolutionary tree and metrics. | ✅ |

---

## 💻 How to Run

### 1. The "Brain" (Training)
*Requires NVIDIA GPU & WSL2 (Linux)*

```bash
# Clone the repository
git clone https://github.com/your-username/EthicaAI.git
cd EthicaAI

# Setup Environment
bash scripts/setup_env.sh

# Start Evolution Loop
bash scripts/run_evolution_gpu.sh
```

### 2. The "Eyes" (Visualization)
*Runs on CPU (Windows/Mac/Linux)*

```bash
# Install Dashboard Dependencies
pip install -r requirements_dashboard.txt

# Launch Dashboard
streamlit run dashboard_evolution.py
```

---

## 🧪 Current Research: RQ-009
> **"Can Inequity Aversion trigger cooperation in a competitive curriculum?"**

- **Hypothesis**: By penalizing wealth disparity, agents might learn to share resources to avoid social penalties.
- **Status**: **Active** (Gen 3+)
- **Theorist's Rationale**: *"Previous competitive strategies failed. Let's introduce a structural 'Envy' mechanism to force fairness."*

---

## 📂 Project Structure

```
EthicaAI/
├── experiments/       # 🧪 Logs, Configs, and Results
├── simulation/        # 🎮 JAX Simulation Core
│   ├── genesis/       # Agents (Theorist, Engineer, Critic)
│   └── jax/           # GPU Kernels
├── dashboard_evolution.py  # 📊 Streamlit Monitor
└── scripts/           # 🛠️ Automation Tools
```

---
**License**: MIT | **Author**: Yesol (Antigravity Agent)

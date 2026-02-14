"""
EthicaAI Experiment Configurations
Defines hyperparameters for small, medium, and large-scale experiments.
All tunable parameters are centralized here (하드코딩 금지 원칙).
"""
import math

# --- HRL Shared Defaults ---
_HRL_DEFAULTS = {
    "HRL_NUM_NEEDS": 2,
    "HRL_NUM_TASKS": 2,
    "HRL_ALPHA": 1.0,
    "HRL_THRESH_INCREASE": 0.005,
    "HRL_THRESH_DECREASE": 0.05,
    "HRL_INTAKE_VAL": 0.2,
    # Environment Reward Scale (Solution A)
    "REWARD_APPLE": 10.0,
    "COST_BEAM": -1.0,
    # Meta-Ranking Parameters (Sen's Optimal Rationality)
    "META_BETA": 0.1,                # ψ self-control cost coefficient
    "META_SURVIVAL_THRESHOLD": -5.0,  # wealth below → λ→0 (survival mode)
    "META_WEALTH_BOOST": 5.0,         # wealth above → λ×1.5 (generosity)
    "META_LAMBDA_EMA": 0.9,           # λ smoothing factor (EMA)
    # Experiment Control Flags
    "USE_META_RANKING": True,          # False = baseline (순수 SVO 보상변환)
    "META_USE_DYNAMIC_LAMBDA": True,   # False = λ = sin(θ) 고정 (ablation)
}

# --- Small Scale (Development & Debugging) ---
CONFIG_SMALL = {
    "ENV_NAME": "cleanup",
    "NUM_AGENTS": 5,
    "ENV_HEIGHT": 25,
    "ENV_WIDTH": 18,
    "MAX_STEPS": 100,
    "NUM_ENVS": 4,
    "NUM_UPDATES": 2,
    "ROLLOUT_LEN": 16,
    "BATCH_SIZE": 4 * 16,
    "LR": 2.5e-4,
    "HIDDEN_DIM": 128,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENTROPY_COEFF": 0.01,
    "VF_COEFF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    **_HRL_DEFAULTS,
}

# --- Medium Scale (Standard Experiments) ---
CONFIG_MEDIUM = {
    "ENV_NAME": "cleanup",
    "NUM_AGENTS": 20,
    "ENV_HEIGHT": 36,
    "ENV_WIDTH": 25,
    "MAX_STEPS": 500,
    "NUM_ENVS": 16,
    "NUM_UPDATES": 300,  # 100→300 for meta-ranking policy differentiation
    "ROLLOUT_LEN": 128,
    "BATCH_SIZE": 16 * 128,
    "LR": 3e-4,
    "HIDDEN_DIM": 128,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENTROPY_COEFF": 0.05,  # Solution D: exploration boost
    "VF_COEFF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    **_HRL_DEFAULTS,
}

# --- Large Scale (Research-Grade) ---
CONFIG_LARGE = {
    "ENV_NAME": "cleanup",
    "NUM_AGENTS": 100,
    "ENV_HEIGHT": 50,
    "ENV_WIDTH": 50,
    "MAX_STEPS": 500,
    "NUM_ENVS": 8,      # 16 → 8 (OOM 방지, 12GB VRAM)
    "NUM_UPDATES": 300,  # 100 → 300 (충분한 학습, Medium과 동일)
    "ROLLOUT_LEN": 128,
    "BATCH_SIZE": 8 * 128,
    "LR": 3e-4,
    "HIDDEN_DIM": 256,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENTROPY_COEFF": 0.01,
    "VF_COEFF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    **_HRL_DEFAULTS,
}

# --- Large Scale Harvest (일반화 검증) ---
CONFIG_LARGE_HARVEST = {
    **CONFIG_LARGE,
    "ENV_NAME": "harvest",  # Cleanup → Harvest 환경
}

# --- SVO Sweep Configurations ---
SVO_SWEEP_THETAS = {
    "selfish": 0.0,                    # 0°
    "individualist": math.pi / 12,     # 15°
    "competitive": math.pi / 6,        # 30°
    "prosocial": math.pi / 4,          # 45° (π/4)
    "cooperative": math.pi / 3,        # 60°
    "altruistic": 5 * math.pi / 12,    # 75°
    "full_altruist": math.pi / 2,      # 90° (π/2)
}

# Utility
def get_config(scale="small"):
    configs = {
        "small": CONFIG_SMALL,
        "medium": CONFIG_MEDIUM,
        "large": CONFIG_LARGE,
        "large_harvest": CONFIG_LARGE_HARVEST,
    }
    return configs.get(scale, CONFIG_SMALL)

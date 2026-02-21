"""
Genesis Experimental Extension for train_pipeline.py
=====================================================
This file contains experimental modifications to the Meta-Ranking logic
for the Genesis (Self-Evolving Research Ecosystem) project.

Usage:
  Import this module and call `apply_genesis_reward()` instead of the
  standard meta-ranking reward calculation in train_pipeline.py.

  To activate, set config["GENESIS_MODE"] = True in config.py.

WARNING: These are EXPERIMENTAL hypotheses. Do NOT merge into the main
         train_pipeline.py without thorough validation.
"""
import jax.numpy as jnp


def apply_genesis_reward(combined_rewards, r_avg_others, lambda_base,
                         flat_levels, config, mode="inverse_beta"):
    """
    Genesis Meta-Ranking 보상 계산 (실험용).

    Args:
        combined_rewards: (B, N) 개별 에이전트 보상
        r_avg_others: (B, N) 타인 평균 보상
        lambda_base: scalar, sin(svo_theta)
        flat_levels: (B*N, NumNeeds) HRL 내부 상태
        config: dict, 실험 설정
        mode: "adaptive_beta" | "inverse_beta" | "institutional"

    Returns:
        meta_ranking_rewards: (B, N)
        debug_info: dict (beta_t, instability 등 디버깅용)
    """
    B, N = combined_rewards.shape

    # 공통: Instability 계산
    reward_gap = jnp.abs(r_avg_others - combined_rewards)  # (B, N)
    instability = reward_gap.mean(axis=-1, keepdims=True)   # (B, 1)

    # --- Hypothesis #001: Adaptive Beta (감쇠형) ---
    if mode == "adaptive_beta":
        beta_base = config.get("GENESIS_BETA_BASE", 10.0)
        gamma = config.get("GENESIS_GAMMA", 2.0)
        beta_t = beta_base * jnp.exp(-gamma * instability)

        # beta_t를 lambda 스케일링에 적용
        adaptive_scale = jnp.exp(-gamma * instability)
        lambda_dynamic = lambda_base * adaptive_scale

    # --- Hypothesis #002: Inverse Adaptive Beta (증폭형) ---
    elif mode == "inverse_beta":
        beta_base = config.get("GENESIS_BETA_BASE", 10.0)
        alpha = config.get("GENESIS_ALPHA", 5.0)
        beta_t = beta_base * (1.0 + alpha * instability)

        # beta_t를 lambda sigmoid에 적용 (핵심: 실제로 사용!)
        lambda_dynamic = jnp.clip(
            lambda_base * jnp.tanh(beta_t * reward_gap.mean(axis=-1, keepdims=True)),
            0.0, 1.0
        )

    # --- Hypothesis #003: Institutional Punishment (미래 확장) ---
    elif mode == "institutional":
        beta_t = jnp.ones((B, 1)) * 10.0
        lambda_dynamic = lambda_base * jnp.ones((B, N))
        # TODO: 비협력자 탐지 + 처벌 보상 체계 구현

    else:
        raise ValueError(f"Unknown genesis mode: {mode}")

    # Wealth-based Dynamic Lambda (기존 로직 유지)
    wealth = flat_levels.mean(axis=-1).reshape((B, N))
    meta_survival = config.get("META_SURVIVAL_THRESHOLD", -5.0)
    meta_boost = config.get("META_WEALTH_BOOST", 5.0)

    lambda_dynamic = jnp.where(
        wealth < meta_survival,
        0.0,
        jnp.where(
            wealth > meta_boost,
            jnp.minimum(1.0, lambda_dynamic * 1.5),
            lambda_dynamic
        )
    )

    # Psi (Self-control cost)
    meta_beta = config.get("META_BETA", 0.1)
    psi = meta_beta * reward_gap

    # Final Reward Mixing
    meta_ranking_rewards = (1 - lambda_dynamic) * combined_rewards + \
                           lambda_dynamic * (r_avg_others - psi)

    debug_info = {
        "beta_t": beta_t,
        "instability": instability,
        "lambda_dynamic": lambda_dynamic,
        "reward_gap": reward_gap,
    }

    return meta_ranking_rewards, debug_info

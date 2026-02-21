
import os
import sys

print("DEBUG: Script started", flush=True)

# 현재 디렉토리를 path에 추가하여 모듈 임포트 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir)) # D:\00.test\PAPER\EthicaAI
sys.path.append(parent_dir)

import jax
# v2.0: GPU 자동 감지 (CPU 강제 제거)
# GPU가 있으면 자동 사용, 없으면 CPU 폴백
print(f"🖥️ JAX Platform: {jax.default_backend()} | Devices: {jax.devices()}")

import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from simulation.jax.config import get_config
from simulation.jax.training.train_pipeline import make_train

def run_genesis_002():
    print("🧬 [Genesis #002] Starting Inverse Beta Experiment...")
    
    # Configuration
    config = get_config("medium") # Medium Scale 사용
    config["GENESIS_MODE"] = True
    config["GENESIS_LOGIC_MODE"] = "inverse_beta"
    config["NUM_AGENTS"] = 100
    config["NUM_ENVS"] = 1   # <-- Max Stability (16 -> 4 -> 1)
    config["BATCH_SIZE"] = 128 # Reduce Batch
    config["NUM_STEPS"] = 200  # 빠른 검증 (500 -> 200) - CPU니까 줄임
    config["LOG_INTERVAL"] = 10
    config["SEEDS"] = [42]  # 일단 1 run만
    
    # Conditions: Prosocial vs Individualist
    svo_conditions = {
        "Prosocial": jnp.pi/4,      # 45도
        "Individualist": 0.0        # 0도 (이기주의자)
    }

    results = []

    train_fn = make_train(config)
    # train_fn = jax.jit(train_fn) # Debugging: Disable JIT
    
    print(f"DEBUG: Config NUM_UPDATES = {config.get('NUM_UPDATES')}")
    print(f"DEBUG: Config NUM_ENVS = {config.get('NUM_ENVS')}")

    for svo_name, svo_val in svo_conditions.items():
        print(f"\n--- Running Condition: {svo_name} (Inverse Beta) ---")
        
        # SVO is passed as argument to train_fn
        
        for seed in config["SEEDS"]:
            print(f"  > Seed {seed}...", end="", flush=True)
            try:
                # Run Simulation
                key = jax.random.PRNGKey(seed)
                # train_fn(rng, svo_theta)
                
                # Check return signature of train: runner_state, metrics_stack
                # metrics_stack is actually metrics dict with history arrays
                runner_state, metrics_history = train_fn(key, float(svo_val))
                
                # Extract Metrics
                # Metrics are arrays over updates. Take mean of last 10 updates for stability
                coop_rate = float(metrics_history["cooperation_rate"][-10:].mean())
                reward_mean = float(metrics_history["reward_mean"][-10:].mean())
                gini = float(metrics_history["gini"][-10:].mean())
                
                results.append({
                    "Mode": "Inverse Beta",
                    "SVO": svo_name,
                    "Seed": seed,
                    "Cooperation": coop_rate,
                    "Reward": reward_mean,
                    "Gini": gini
                })
                print(f" Done. Coop={coop_rate:.4f}")
                
            except Exception as e:
                print(f" Failed: {e}")
                import traceback
                traceback.print_exc()

    # Analysis
    df = pd.DataFrame(results)
    print("\n\n📊 [Genesis #002] Experiment Results Summary:")
    print(df.groupby(["Mode", "SVO"]).mean(numeric_only=True))
    
    # Save Report
    report_path = os.path.join(parent_dir, "simulation", "outputs", "genesis_002_report.csv")
    df.to_csv(report_path, index=False)
    print(f"\nSaved report to {report_path}")

if __name__ == "__main__":
    run_genesis_002()

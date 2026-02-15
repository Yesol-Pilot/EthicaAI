import os
import sys
import time

# Add current directory to path
sys.path.append(os.getcwd())

from simulation.jax.experiment_jax import run_sweep
from simulation.jax.config import SVO_SWEEP_THETAS

def run_genesis_experiment_002():
    print(">>> Starting Genesis Run #002 (Hypothesis: Crisis Response) <<<")
    # 검증을 위해 'Full Altruist' 조건에서 테스트
    genesis_angles = {
        "genesis_crisis_response": SVO_SWEEP_THETAS["full_altruist"]
    }
    
    output_dir = "simulation/outputs/genesis_002"
    os.makedirs(output_dir, exist_ok=True)
    
    results, path = run_sweep(
        scale="small",
        svo_angles=genesis_angles,
        seeds=[1004], # Same Angel Seed
        output_dir=output_dir
    )
    
    print(f"Genesis Run #002 Complete. Results at {path}")

if __name__ == "__main__":
    run_genesis_experiment_002()

#!/usr/bin/env python3
"""
EthicaAI — CPU/메모리 집약적 분석 병렬 실행기.

GPU가 Full Sweep 학습에 쓰이는 동안, CPU와 RAM으로 기존 데이터 분석을 병렬 수행합니다.
모든 결과는 experiments/cpu_analysis_results/ 에 저장됩니다.

실행:
    source ~/ethicaai_env/bin/activate
    cd /mnt/d/00.test/PAPER/EthicaAI
    python3 scripts/cpu_heavy_analysis.py 2>&1 | tee experiments/cpu_analysis_log.txt
"""

import os
import sys
import json
import time
import traceback
import logging
import multiprocessing as mp
from datetime import datetime

# 프로젝트 루트
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("CPUAnalysis")

OUTPUT_DIR = os.path.join(project_root, "experiments", "cpu_analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# JAX가 GPU를 잡지 않도록 설정 (CPU 전용)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def task_lmm_causal_forest():
    """LMM + Causal Forest 분석 (CPU/RAM 집약적 - 약 15~30분)."""
    log.info("🔬 [1/5] LMM + Causal Forest 분석 시작...")
    start = time.time()
    
    try:
        from simulation.jax.analysis.lmm_causal_forest import (
            generate_panel_data, lmm_analysis, causal_forest_simulation,
            plot_fig51, plot_fig52
        )
        
        # 100 에이전트 × 10 시드 × 200 스텝 패널 데이터 생성
        data = generate_panel_data()
        log.info(f"  패널 데이터 생성 완료: {len(data)} 행")
        
        # LMM 분석
        lmm_results = lmm_analysis(data)
        log.info("  LMM 분석 완료")
        
        # Causal Forest
        agent_features, hte_by_svo = causal_forest_simulation(data)
        log.info("  Causal Forest 완료")
        
        # 시각화
        try:
            plot_fig51(lmm_results)
            plot_fig52(agent_features, hte_by_svo)
            log.info("  시각화 완료")
        except Exception as e:
            log.warning(f"  시각화 실패 (헤드리스?): {e}")
        
        # 결과 저장
        result = {
            "lmm": lmm_results,
            "hte_svo_keys": list(hte_by_svo.keys()) if hte_by_svo else [],
            "n_agents": len(agent_features) if agent_features is not None else 0,
        }
        
        elapsed = time.time() - start
        log.info(f"  ✅ LMM/Causal Forest 완료 ({elapsed:.0f}초)")
        return result
        
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"  ❌ LMM/Causal Forest 실패: {e} ({elapsed:.0f}초)")
        traceback.print_exc()
        return {"error": str(e)}


def task_sensitivity_analysis():
    """민감도 분석 (CPU 집약적)."""
    log.info("📊 [2/5] 민감도 분석 시작...")
    start = time.time()
    
    try:
        from simulation.jax.analysis.sensitivity_analysis import main as sensitivity_main
        sensitivity_main()
        elapsed = time.time() - start
        log.info(f"  ✅ 민감도 분석 완료 ({elapsed:.0f}초)")
        return {"status": "completed", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"  ❌ 민감도 분석 실패: {e} ({elapsed:.0f}초)")
        return {"error": str(e)}


def task_lyapunov_analysis():
    """리아푸노프 안정성 분석 (CPU 집약적)."""
    log.info("🔢 [3/5] 리아푸노프 분석 시작...")
    start = time.time()
    
    try:
        from simulation.jax.analysis.lyapunov_analysis import main as lyapunov_main
        lyapunov_main()
        elapsed = time.time() - start
        log.info(f"  ✅ 리아푸노프 분석 완료 ({elapsed:.0f}초)")
        return {"status": "completed", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"  ❌ 리아푸노프 분석 실패: {e} ({elapsed:.0f}초)")
        return {"error": str(e)}


def task_convergence_proof():
    """수렴 증명 시뮬레이션 (수학적 검증, CPU 집약적)."""
    log.info("📐 [4/5] 수렴 증명 시뮬레이션 시작...")
    start = time.time()
    
    try:
        from simulation.jax.analysis.convergence_proof import main as convergence_main
        convergence_main()
        elapsed = time.time() - start
        log.info(f"  ✅ 수렴 증명 완료 ({elapsed:.0f}초)")
        return {"status": "completed", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"  ❌ 수렴 증명 실패: {e} ({elapsed:.0f}초)")
        return {"error": str(e)}


def task_scale_analysis():
    """대규모 스케일 비교 분석 (메모리 집약적 - 1000 에이전트)."""
    log.info("📈 [5/5] 스케일 비교 분석 시작...")
    start = time.time()
    
    try:
        from simulation.jax.analysis.scale_comparison import main as scale_main
        scale_main()
        elapsed = time.time() - start
        log.info(f"  ✅ 스케일 비교 완료 ({elapsed:.0f}초)")
        return {"status": "completed", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"  ❌ 스케일 비교 실패: {e} ({elapsed:.0f}초)")
        return {"error": str(e)}


def task_bootstrap_on_existing_data():
    """기존 실험 데이터에 대한 Bootstrap CI 분석 (CPU 집약적)."""
    log.info("🎲 [BONUS] 기존 데이터 Bootstrap CI 분석 시작...")
    start = time.time()
    
    try:
        import numpy as np
        from simulation.jax.analysis.bootstrap_ci import bootstrap_ate
        
        # 기존 sweep 결과 로드
        data_dir = os.path.join(project_root, "simulation", "outputs", "reproduce")
        sweep_file = os.path.join(data_dir, "full_sweep_results.json")
        
        if not os.path.exists(sweep_file):
            log.warning("  기존 sweep 데이터 없음, 스킵")
            return {"status": "skipped", "reason": "no existing sweep data"}
        
        with open(sweep_file, "r") as f:
            sweep_data = json.load(f)
        
        # 데이터 추출
        thetas = []
        rewards = []
        coops = []
        
        for condition_name, condition_data in sweep_data.items():
            if isinstance(condition_data, dict) and "runs" in condition_data:
                theta = condition_data.get("theta", 0.0)
                for run in condition_data["runs"]:
                    metrics = run.get("metrics", {})
                    thetas.append(theta)
                    reward_series = metrics.get("reward_mean", [0])
                    coop_series = metrics.get("cooperation_rate", [0])
                    rewards.append(reward_series[-1] if reward_series else 0)
                    coops.append(coop_series[-1] if coop_series else 0)
        
        if len(thetas) < 5:
            log.warning(f"  데이터 부족: {len(thetas)}개 포인트")
            return {"status": "insufficient_data", "n_points": len(thetas)}
        
        T = np.array(thetas)
        
        # 50,000회 부트스트랩 (CPU 고강도)
        log.info(f"  {len(thetas)}개 데이터포인트에 50,000회 부트스트랩 실행 중...")
        
        results = {
            "reward_ate": bootstrap_ate(T, np.array(rewards), n_bootstrap=50000),
            "cooperation_ate": bootstrap_ate(T, np.array(coops), n_bootstrap=50000),
        }
        
        # 결과 저장
        output_file = os.path.join(OUTPUT_DIR, "bootstrap_ci_existing_data.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start
        log.info(f"  ✅ Bootstrap CI 완료 ({elapsed:.0f}초)")
        log.info(f"    Reward ATE: {results['reward_ate']['ate_mean']:.6f} "
                f"CI: [{results['reward_ate']['ci_lower']:.6f}, {results['reward_ate']['ci_upper']:.6f}]")
        log.info(f"    Coop ATE: {results['cooperation_ate']['ate_mean']:.6f} "
                f"CI: [{results['cooperation_ate']['ci_lower']:.6f}, {results['cooperation_ate']['ci_upper']:.6f}]")
        
        return results
        
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"  ❌ Bootstrap CI 실패: {e} ({elapsed:.0f}초)")
        traceback.print_exc()
        return {"error": str(e)}


def task_paper_figures():
    """논문 Figure 일괄 생성 (CPU + 디스크 집약적)."""
    log.info("🖼️ [BONUS] 논문 Figure 일괄 생성 시작...")
    start = time.time()
    
    try:
        from simulation.jax.analysis.paper_figures import main as figures_main
        figures_main()
        elapsed = time.time() - start
        log.info(f"  ✅ Figure 생성 완료 ({elapsed:.0f}초)")
        return {"status": "completed", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start
        log.error(f"  ❌ Figure 생성 실패: {e} ({elapsed:.0f}초)")
        return {"error": str(e)}


def main():
    """CPU/메모리 집약적 분석 순차 실행."""
    start_time = datetime.now()
    
    log.info("=" * 60)
    log.info("EthicaAI — CPU/메모리 집약적 분석 시작")
    log.info(f"시작 시각: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"CPU 코어: {mp.cpu_count()}")
    log.info(f"JAX 플랫폼: CPU (GPU는 Full Sweep에 사용 중)")
    log.info(f"출력 디렉토리: {OUTPUT_DIR}")
    log.info("=" * 60)
    
    all_results = {}
    
    # 순차 실행 (메모리 충돌 방지)
    tasks = [
        ("bootstrap_ci", task_bootstrap_on_existing_data),
        ("lmm_causal_forest", task_lmm_causal_forest),
        ("sensitivity", task_sensitivity_analysis),
        ("lyapunov", task_lyapunov_analysis),
        ("convergence", task_convergence_proof),
        ("scale_comparison", task_scale_analysis),
        ("paper_figures", task_paper_figures),
    ]
    
    for task_name, task_fn in tasks:
        log.info(f"\n{'='*40}")
        log.info(f"작업: {task_name}")
        log.info(f"{'='*40}")
        
        try:
            result = task_fn()
            all_results[task_name] = result
            
            # 중간 저장
            task_file = os.path.join(OUTPUT_DIR, f"{task_name}.json")
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            log.info(f"💾 저장: {task_file}")
            
        except Exception as e:
            log.error(f"❌ {task_name} 전체 실패: {e}")
            all_results[task_name] = {"error": str(e)}
    
    # 전체 요약 저장
    end_time = datetime.now()
    summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_elapsed": str(end_time - start_time),
        "results": all_results,
    }
    
    summary_file = os.path.join(OUTPUT_DIR, "cpu_analysis_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    log.info("\n" + "=" * 60)
    log.info("🏁 CPU 분석 완료!")
    log.info(f"소요: {end_time - start_time}")
    log.info(f"결과: {summary_file}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

"""
EthicaAI 재현성 스크립트 (reproduce.py)
NeurIPS 2026

이 스크립트 하나로 논문의 모든 분석 결과를 재현합니다.

사용법:
  python reproduce.py              # 전체 파이프라인
  python reproduce.py --phase G    # Phase G만
  python reproduce.py --phase H    # Phase H만
  python reproduce.py --quick      # 빠른 데모 (시드 1개)
"""
import os
import sys
import time
import argparse
import subprocess

# 출력 디렉토리 설정
OUTPUT_DIR = os.environ.get("ETHICAAI_OUTPUT_DIR", "simulation/outputs/reproduce")

# 분석 모듈 목록 (순서대로 실행)
ANALYSES = {
    "G2": {
        "name": "수렴성 증명 (Convergence Proof)",
        "module": "simulation.jax.analysis.convergence_proof",
        "phase": "G",
    },
    "G3": {
        "name": "Static vs Dynamic Lambda",
        "module": "simulation.jax.analysis.static_vs_dynamic",
        "phase": "G",
    },
    "G1": {
        "name": "파라미터 민감도 분석 (Sensitivity)",
        "module": "simulation.jax.analysis.sensitivity_analysis",
        "phase": "G",
    },
    "G4": {
        "name": "Cross-Environment (IPD)",
        "module": "simulation.jax.analysis.cross_env_validation",
        "phase": "G",
    },
    "G5": {
        "name": "N-Player PGG Experiment",
        "module": "simulation.jax.analysis.pgg_experiment",
        "phase": "G",
    },
    "H1": {
        "name": "진화적 경쟁 시뮬레이션 (Evolutionary Competition)",
        "module": "simulation.jax.analysis.evolutionary_competition",
        "phase": "H",
    },
    "H2": {
        "name": "메커니즘 분해 (Mechanism Decomposition)",
        "module": "simulation.jax.analysis.mechanism_decomposition",
        "phase": "H",
    },
    "M1": {
        "name": "Full Sweep (4환경 × 7SVO × 10seeds)",
        "module": "simulation.jax.analysis.run_full_sweep",
        "phase": "M",
    },
    "M2": {
        "name": "Mixed-SVO Population (임계점 분석)",
        "module": "simulation.jax.analysis.mixed_svo_experiment",
        "phase": "M",
    },
    "M3": {
        "name": "Communication Channels (Cheap Talk)",
        "module": "simulation.jax.analysis.communication_experiment",
        "phase": "M",
    },
    "M4": {
        "name": "Continuous PGG (연속 행동 공간)",
        "module": "simulation.jax.analysis.continuous_experiment",
        "phase": "M",
    },
    "N1": {
        "name": "MAPPO 멀티 환경 훈련 시뮬레이션",
        "module": "simulation.jax.analysis.mappo_training_sim",
        "phase": "N",
    },
    "N2": {
        "name": "Partial Observability (정보 비대칭)",
        "module": "simulation.jax.analysis.partial_obs_experiment",
        "phase": "N",
    },
    "N3": {
        "name": "Multi-Resource (2-자원 PGG)",
        "module": "simulation.jax.analysis.multi_resource_experiment",
        "phase": "N",
    },
    "N4": {
        "name": "LLM vs λ 비교 (Constitutional)",
        "module": "simulation.jax.analysis.llm_comparison_experiment",
        "phase": "N",
    },
}


def run_analysis(key, info, output_dir):
    """단일 분석 모듈 실행."""
    print(f"\n{'='*60}")
    print(f"  [{key}] {info['name']}")
    print(f"{'='*60}")
    
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    result = subprocess.run(
        [sys.executable, "-m", info["module"], output_dir],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    
    elapsed = time.time() - t0
    
    if result.returncode == 0:
        print(result.stdout)
        print(f"  ✓ {key} 완료 ({elapsed:.1f}초)")
        return True
    else:
        print(f"  ✗ {key} 실패!")
        print(f"  STDERR: {result.stderr[:500]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="EthicaAI 재현성 스크립트")
    parser.add_argument("--phase", choices=["G", "H", "M", "N", "all"], default="all",
                       help="실행할 Phase (기본: all)")
    parser.add_argument("--quick", action="store_true",
                       help="빠른 데모 모드 (축소 실행)")
    parser.add_argument("--output", default=OUTPUT_DIR,
                       help="출력 디렉토리")
    args = parser.parse_args()
    
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  EthicaAI Reproduction Pipeline")
    print(f"  Phase: {args.phase}")
    print(f"  Output: {output_dir}")
    print(f"  Mode: {'Quick Demo' if args.quick else 'Full'}")
    print("=" * 60)
    
    # 실행 대상 필터링
    targets = {}
    for key, info in ANALYSES.items():
        if args.phase == "all" or info["phase"] == args.phase:
            targets[key] = info
    
    results = {}
    total_start = time.time()
    
    for key, info in targets.items():
        success = run_analysis(key, info, output_dir)
        results[key] = success
    
    total_time = time.time() - total_start
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("  REPRODUCTION SUMMARY")
    print("=" * 60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for key, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {key}: {ANALYSES[key]['name']}")
    
    print(f"\n  Total: {success_count}/{total_count} succeeded")
    print(f"  Time: {total_time:.1f}초")
    
    if success_count == total_count:
        print("\n  🎉 전체 재현 성공!")
    else:
        print("\n  ⚠ 일부 실패 — 로그를 확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()

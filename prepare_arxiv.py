"""
EthicaAI M6: arXiv 프리프린트 제출 패키지 준비
- LaTeX 소스 + Figure 번들링
- 참고문헌 형식 검증
- .tar.gz 제출 패키지 생성
"""
import os
import sys
import shutil
import tarfile
import json
from pathlib import Path

# ── 경로 설정 ──
PROJECT_ROOT = Path(__file__).parent.resolve()
SUBMISSION_DIR = PROJECT_ROOT / "submission_arxiv"
FIGURES_DIR = PROJECT_ROOT / "simulation" / "outputs" / "reproduce"
SITE_FIGURES_DIR = PROJECT_ROOT / "site" / "figures"
TEX_DIR = PROJECT_ROOT / "submission_neurips"

# arXiv 카테고리 및 메타데이터
ARXIV_METADATA = {
    "title": "Testing Amartya Sen's Optimal Rationality Theory "
             "through Multi-Agent Reinforcement Learning: "
             "When Should AI Agents Be Moral?",
    "authors": ["Yesol Heo"],
    "primary_category": "cs.AI",
    "cross_list": ["cs.MA", "cs.GT"],
    "license": "CC BY 4.0",
    "abstract_file": "abstract.txt",
}

# 필수 Figure 파일 목록 (30개)
REQUIRED_FIGURES = [
    f"fig{i}_{name}.png" for i, name in [
        (1, "learning_curves"), (2, "cooperation_rate"), (3, "threshold_evolution"),
        (4, "lambda_dynamics"), (5, "gini_comparison"), (6, "causal_forest"),
        (7, "svo_vs_welfare"), (8, "summary_heatmap"), (9, "role_specialization"),
        (10, "scale_comparison"), (11, "ate_scale_comparison"),
        (12, "convergence"), (13, "static_dynamic"),
        (14, "sensitivity"), (15, "ipd_cross"),
        (16, "pgg_experiment"), (17, "evolutionary"),
        (18, "mechanism"), (19, "human_comparison"),
        (20, "harvest_cross"), (21, "scale_1000"),
        (22, "melting_pot"), (23, "constitutional"),
        (24, "full_sweep"), (25, "mixed_svo"),
        (26, "mixed_ate"), (27, "communication"),
        (28, "truthfulness"), (29, "continuous_dist"),
        (30, "continuous_lambda"),
    ]
]


def prepare_submission():
    """arXiv 제출 패키지 준비."""
    print("[M6] arXiv 제출 패키지 준비 시작...")
    
    # 1. 제출 디렉토리 생성
    if SUBMISSION_DIR.exists():
        shutil.rmtree(SUBMISSION_DIR)
    SUBMISSION_DIR.mkdir(parents=True)
    figures_out = SUBMISSION_DIR / "figures"
    figures_out.mkdir()
    
    # 2. Figure 수집 (site/figures → submission_arxiv/figures)
    found = 0
    missing = []
    for fig_name in REQUIRED_FIGURES:
        src = SITE_FIGURES_DIR / fig_name
        if not src.exists():
            # Fallback: reproduce 폴더
            src = FIGURES_DIR / fig_name
        if src.exists():
            shutil.copy2(src, figures_out / fig_name)
            found += 1
        else:
            missing.append(fig_name)
    
    print(f"  Figure: {found}/{len(REQUIRED_FIGURES)} 수집 완료")
    if missing:
        print(f"  ⚠ 누락: {missing[:5]}...")
    
    # 3. LaTeX 소스 복사
    if TEX_DIR.exists():
        for f in TEX_DIR.glob("*.tex"):
            shutil.copy2(f, SUBMISSION_DIR)
        for f in TEX_DIR.glob("*.bib"):
            shutil.copy2(f, SUBMISSION_DIR)
        for f in TEX_DIR.glob("*.sty"):
            shutil.copy2(f, SUBMISSION_DIR)
        for f in TEX_DIR.glob("*.bst"):
            shutil.copy2(f, SUBMISSION_DIR)
        print(f"  LaTeX: {len(list(SUBMISSION_DIR.glob('*.tex')))} .tex 파일 복사")
    else:
        print("  ⚠ submission_neurips 디렉토리 없음 — LaTeX 소스 누락")
    
    # 4. Abstract 파일 생성
    abstract = (
        "Integrating AI agents into human society requires resolving the fundamental "
        "conflict between self-interest and social values. This study formalizes Amartya "
        "Sen's Meta-Ranking theory—preferences over preferences—within a Multi-Agent "
        "Reinforcement Learning (MARL) framework. We simulate agents with 7 Social Value "
        "Orientations (SVOs) across 4 environments (Cleanup, IPD, PGG, Harvest) at scales "
        "up to 1,000 agents.\n\n"
        "Five key findings: (1) Dynamic meta-ranking significantly improves collective "
        "welfare (p=0.0003) while static SVO injection fails; (2) Agents exhibit spontaneous "
        "role specialization; (3) Only 'Situational Commitment' survives as an ESS, converging "
        "to ~12% of the population; (4) Individualist SVO (theta=15°) best matches human "
        "behavioral data (WD=0.053); (5) Full factorial decomposition shows SVO rotation accounts "
        "for 86% of the effect.\n\n"
        "Extended experiments demonstrate generalizability across 4 environments (560 runs), "
        "tipping points in mixed-motive populations (~30%), communication-enhanced cooperation "
        "(+5.8%), and continuous action space robustness (ATE≈+0.20). These results provide "
        "design principles for AI systems that can promote the common good without becoming "
        "'suckers' — computationally realizing morality as an Evolutionarily Stable Strategy."
    )
    
    abstract_path = SUBMISSION_DIR / ARXIV_METADATA["abstract_file"]
    with open(abstract_path, 'w', encoding='utf-8') as f:
        f.write(abstract)
    print(f"  Abstract: {abstract_path}")
    
    # 5. 메타데이터 저장
    meta_path = SUBMISSION_DIR / "arxiv_metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(ARXIV_METADATA, f, indent=2, ensure_ascii=False)
    print(f"  메타데이터: {meta_path}")
    
    # 6. README
    readme_content = (
        "# EthicaAI — arXiv Submission Package\n\n"
        f"- Primary: {ARXIV_METADATA['primary_category']}\n"
        f"- Cross-list: {', '.join(ARXIV_METADATA['cross_list'])}\n"
        f"- License: {ARXIV_METADATA['license']}\n"
        f"- Figures: {found}\n\n"
        "## Build\n"
        "```\npdflatex main.tex\nbibtex main\npdflatex main.tex\npdflatex main.tex\n```\n"
    )
    with open(SUBMISSION_DIR / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    # 7. .tar.gz 패키지 생성
    tar_path = PROJECT_ROOT / "ethicaai_arxiv.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in SUBMISSION_DIR.rglob("*"):
            if f.is_file():
                arcname = f"ethicaai/{f.relative_to(SUBMISSION_DIR)}"
                tar.add(f, arcname=arcname)
    
    tar_size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print(f"\n  📦 arXiv 패키지: {tar_path}")
    print(f"  📏 패키지 크기: {tar_size_mb:.1f} MB")
    
    # 8. 검증 요약
    print("\n" + "=" * 60)
    print("M6 arXiv SUBMISSION READINESS")
    print("=" * 60)
    print(f"  Title: {ARXIV_METADATA['title'][:60]}...")
    print(f"  Authors: {', '.join(ARXIV_METADATA['authors'])}")
    print(f"  Categories: {ARXIV_METADATA['primary_category']} "
          f"[{', '.join(ARXIV_METADATA['cross_list'])}]")
    print(f"  Figures: {found}/{len(REQUIRED_FIGURES)}")
    print(f"  Package: {tar_size_mb:.1f} MB")
    
    if found >= 20:
        print("\n  ✅ arXiv 제출 준비 완료!")
    else:
        print(f"\n  ⚠ Figure {len(REQUIRED_FIGURES) - found}개 누락 — 확인 필요")
    
    return str(tar_path)


if __name__ == "__main__":
    prepare_submission()

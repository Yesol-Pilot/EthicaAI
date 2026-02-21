"""
EthicaAI Zenodo Upload Script
- HTML → PDF 변환 (Playwright)
- 기존 파일 삭제 + 새 PDF 업로드
- 메타데이터 업데이트 + 퍼블리시

Usage:
    python scripts/zenodo_upload.py              # PDF 변환 + 업로드 (퍼블리시 안 함)
    python scripts/zenodo_upload.py --publish     # PDF 변환 + 업로드 + 퍼블리시
    python scripts/zenodo_upload.py --pdf-only    # PDF 변환만
    python scripts/zenodo_upload.py --upload-only # 기존 PDF 업로드만
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv

# ===== Constants =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
HTML_PATH = PROJECT_ROOT / "submission" / "paper_english_v2.html"
PDF_PATH = PROJECT_ROOT / "submission" / "paper_english_v2.pdf"
PDF_FILENAME = "Computational Verification of Amartya Sen's Optimal Rationality via Multi-Agent Reinforcement Learning with Meta-Ranking.pdf"

ZENODO_API_BASE = "https://zenodo.org/api"


def load_config():
    """Load environment variables."""
    load_dotenv(ENV_PATH)
    token = os.getenv("ZENODO_ACCESS_TOKEN")
    record_id = os.getenv("ZENODO_RECORD_ID")
    if not token:
        print("❌ ZENODO_ACCESS_TOKEN not found in .env")
        sys.exit(1)
    return token, record_id


def convert_html_to_pdf():
    """Convert HTML to PDF using Playwright."""
    print(f"\n📄 HTML → PDF 변환 중...")
    print(f"   소스: {HTML_PATH}")
    print(f"   출력: {PDF_PATH}")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("❌ Playwright가 설치되지 않았습니다.")
        print("   pip install playwright; playwright install chromium")
        sys.exit(1)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        # Load HTML file
        file_url = HTML_PATH.as_uri()
        page.goto(file_url, wait_until="networkidle")

        # Wait for MathJax rendering
        print("   ⏳ MathJax 렌더링 대기 (5초)...")
        page.wait_for_timeout(5000)

        # Generate PDF
        page.pdf(
            path=str(PDF_PATH),
            format="A4",
            margin={"top": "15mm", "bottom": "15mm", "left": "18mm", "right": "18mm"},
            print_background=True,
            prefer_css_page_size=True,
        )
        browser.close()

    size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
    print(f"   ✅ PDF 생성 완료: {size_mb:.1f} MB")
    return PDF_PATH


def get_draft_deposition(token):
    """Find existing draft or create new version."""
    headers = {"Authorization": f"Bearer {token}"}

    # List depositions to find draft
    resp = requests.get(f"{ZENODO_API_BASE}/deposit/depositions", headers=headers)
    resp.raise_for_status()
    depositions = resp.json()

    # Find draft (state = 'inprogress')
    drafts = [d for d in depositions if d.get("state") == "inprogress"]

    if drafts:
        draft = drafts[0]
        print(f"   📝 기존 초안 발견: #{draft['id']}")
        return draft

    # No draft found - create new version from latest published
    record_id = os.getenv("ZENODO_RECORD_ID")
    if not record_id:
        print("❌ No draft found and ZENODO_RECORD_ID not set")
        sys.exit(1)

    print(f"   🔨 새 버전 생성 중 (from #{record_id})...")
    resp = requests.post(
        f"{ZENODO_API_BASE}/deposit/depositions/{record_id}/actions/newversion",
        headers=headers,
    )
    resp.raise_for_status()
    new_version_url = resp.json()["links"]["latest_draft"]
    resp2 = requests.get(new_version_url, headers=headers)
    resp2.raise_for_status()
    draft = resp2.json()
    print(f"   ✅ 새 버전 초안 생성: #{draft['id']}")
    return draft


def upload_pdf(token, draft, pdf_path):
    """Delete old files and upload new PDF."""
    deposition_id = draft["id"]
    headers = {"Authorization": f"Bearer {token}"}

    # Delete existing files
    print("\n🗑️  기존 파일 삭제 중...")
    files_resp = requests.get(
        f"{ZENODO_API_BASE}/deposit/depositions/{deposition_id}/files",
        headers=headers,
    )
    files_resp.raise_for_status()
    for f in files_resp.json():
        file_id = f["id"]
        requests.delete(
            f"{ZENODO_API_BASE}/deposit/depositions/{deposition_id}/files/{file_id}",
            headers=headers,
        )
        print(f"   삭제됨: {f['filename']}")

    # Upload new PDF
    print(f"\n📤 새 PDF 업로드 중...")
    bucket_url = draft["links"]["bucket"]

    with open(pdf_path, "rb") as fp:
        resp = requests.put(
            f"{bucket_url}/{PDF_FILENAME}",
            data=fp,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream",
            },
        )
    resp.raise_for_status()
    result = resp.json()
    size_mb = result["size"] / (1024 * 1024)
    print(f"   ✅ 업로드 완료: {PDF_FILENAME} ({size_mb:.1f} MB)")
    return result


def update_metadata(token, draft):
    """Update deposition metadata for v2."""
    deposition_id = draft["id"]
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    metadata = {
        "metadata": {
            "title": "Computational Verification of Amartya Sen's Optimal Rationality via Multi-Agent Reinforcement Learning with Meta-Ranking",
            "upload_type": "publication",
            "publication_type": "preprint",
            "description": (
                "<p>This study formalizes Amartya Sen's Meta-Ranking theory within a "
                "Multi-Agent Reinforcement Learning (MARL) framework, demonstrating that "
                "dynamic moral commitment contingent on resource states—rather than simple "
                "social preference injection—is the key mechanism for resolving social dilemmas.</p>"
                "<p><strong>v2 Updates (2026-02-21):</strong> Full experimental results from "
                "Stages 2-7 including 560+ runs across 4 environments, 100-agent scalability "
                "verification, cross-environment validation (IPD, PGG, Harvest), human-AI "
                "behavioral alignment (WD<0.2), evolutionary stability analysis, Byzantine "
                "robustness, network topology effects, and policy implications. Total 60 figures.</p>"
                "<p>Code: https://github.com/Yesol-Pilot/EthicaAI</p>"
            ),
            "creators": [{"name": "Heo, Yesol", "affiliation": "Independent Researcher"}],
            "keywords": [
                "Meta-Ranking",
                "Social Value Orientation",
                "Causal Inference",
                "AI Alignment",
                "Amartya Sen",
                "Multi-Agent Reinforcement Learning",
                "Evolutionary Stability",
            ],
            "access_right": "open",
            "license": "cc-by-4.0",
            "publication_date": "2026-02-21",
            "language": "eng",
        }
    }

    print("\n📋 메타데이터 업데이트 중...")
    resp = requests.put(
        f"{ZENODO_API_BASE}/deposit/depositions/{deposition_id}",
        data=json.dumps(metadata),
        headers=headers,
    )
    resp.raise_for_status()
    print("   ✅ 메타데이터 업데이트 완료")
    return resp.json()


def publish(token, draft):
    """Publish the deposition."""
    deposition_id = draft["id"]
    headers = {"Authorization": f"Bearer {token}"}

    print("\n🚀 퍼블리시 중...")
    resp = requests.post(
        f"{ZENODO_API_BASE}/deposit/depositions/{deposition_id}/actions/publish",
        headers=headers,
    )
    resp.raise_for_status()
    result = resp.json()
    doi = result.get("doi", "N/A")
    print(f"   ✅ 퍼블리시 완료!")
    print(f"   DOI: {doi}")
    print(f"   URL: https://zenodo.org/records/{result['id']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="EthicaAI Zenodo Upload")
    parser.add_argument("--publish", action="store_true", help="Publish after upload")
    parser.add_argument("--pdf-only", action="store_true", help="Generate PDF only")
    parser.add_argument("--upload-only", action="store_true", help="Upload existing PDF only")
    args = parser.parse_args()

    token, record_id = load_config()

    print("=" * 60)
    print("EthicaAI Zenodo Upload Script")
    print("=" * 60)

    # Step 1: PDF conversion
    if not args.upload_only:
        pdf_path = convert_html_to_pdf()
    else:
        if not PDF_PATH.exists():
            print(f"❌ PDF not found: {PDF_PATH}")
            sys.exit(1)
        pdf_path = PDF_PATH
        print(f"📄 기존 PDF 사용: {pdf_path}")

    if args.pdf_only:
        print("\n✅ PDF 생성 완료. --pdf-only 모드로 종료합니다.")
        return

    # Step 2: Get/Create draft
    print("\n🔍 Zenodo 초안 확인 중...")
    draft = get_draft_deposition(token)

    # Step 3: Upload PDF
    upload_pdf(token, draft, pdf_path)

    # Step 4: Update metadata
    update_metadata(token, draft)

    # Step 5: Publish (optional)
    if args.publish:
        publish(token, draft)
    else:
        print(f"\n⚠️  퍼블리시하지 않았습니다. 확인 후 아래 명령으로 퍼블리시:")
        print(f"   python scripts/zenodo_upload.py --upload-only --publish")
        print(f"   또는 Zenodo 웹에서 직접 Publish")

    print("\n" + "=" * 60)
    print("완료!")


if __name__ == "__main__":
    main()

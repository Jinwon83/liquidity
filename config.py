"""프로젝트 공통 설정."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# Windows에서 .env 가 cp949로 저장된 경우도 허용
_env = ROOT / ".env"
if _env.exists():
    try:
        load_dotenv(_env, encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(_env, encoding="cp949")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual"

DEFAULT_START = "2025-08-01"
DEFAULT_END = "2026-07-22"

FRED_SERIES_IDS = [
    "WALCL",
    "RRPONTSYD",
    "WTREGEN",
    "SOFR",
    "IORB",
    "DGS10",
    "DTWEXBGS",
    "BAMLH0A0HYM2",
]

YFINANCE_TICKERS = [
    "^KS11",
    "^KQ11",
    "^IXIC",
    "^GSPC",
    "^VIX",
    "DX-Y.NYB",  # ICE Dollar Index (DXY) — DTWEXBGS와 별개
    "EEM",  # iShares MSCI Emerging Markets ETF
    "000660.KS",
    "7709.HK",
]

# yfinance 티커 → 컬럼 prefix 매핑 (특수문자 정리)
YFINANCE_COL_PREFIX = {
    "^KS11": "KS11",
    "^KQ11": "KQ11",
    "^IXIC": "IXIC",
    "^GSPC": "GSPC",
    "^VIX": "VIX",
    "DX-Y.NYB": "DXY",
    "EEM": "EEM",
    "000660.KS": "000660_KS",
    "7709.HK": "7709_HK",
}

# LS t8429 업종차트 — VKOSPI upcode
VKOSPI_UPCODE = "205"

# 한국은행 기준금리 (일별)
ECOS_BASE_RATE_STAT = "722Y001"
ECOS_BASE_RATE_ITEM = "0101000"

# LS증권 경로 (토큰/설정 재사용)
LS_DIR = Path(os.getenv("LS_DIR", str(ROOT.parent / "LS증권")))

FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
ECOS_API_KEY = os.getenv("ECOS_API_KEY", "").strip()
LS_APP_KEY = os.getenv("LS_APP_KEY", "").strip()
LS_APP_SECRET_KEY = os.getenv("LS_APP_SECRET_KEY", "").strip()
KRX_JSESSIONID = os.getenv("KRX_JSESSIONID", "").strip()
KRX_AUTH_TOKEN = os.getenv("KRX_AUTH_TOKEN", "").strip()


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)

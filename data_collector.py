"""
KOSPI-NASDAQ / 달러 유동성 가설용 데이터 수집 모듈.

소스별 함수가 독립적으로 동작하며, 한 소스 실패가 전체를 멈추지 않는다.
성공한 raw 결과는 data/raw/{source}_{YYYYMMDD}.csv 로 저장한다.
"""
from __future__ import annotations

import io
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

import config as project_config
from config import (
    DEFAULT_END,
    DEFAULT_START,
    ECOS_API_KEY,
    ECOS_BASE_RATE_ITEM,
    ECOS_BASE_RATE_STAT,
    FRED_API_KEY,
    FRED_SERIES_IDS,
    KRX_AUTH_TOKEN,
    KRX_JSESSIONID,
    LS_APP_KEY,
    LS_APP_SECRET_KEY,
    LS_DIR,
    MANUAL_DIR,
    RAW_DIR,
    VKOSPI_UPCODE,
    YFINANCE_COL_PREFIX,
    YFINANCE_TICKERS,
    ensure_dirs,
)

# Windows 콘솔 UTF-8
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def _today_tag() -> str:
    return datetime.now().strftime("%Y%m%d")


def _to_yyyymmdd(value: str | datetime | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _to_iso(value: str | datetime | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def save_raw_csv(df: pd.DataFrame, source_name: str, tag: str | None = None) -> Path | None:
    """DataFrame을 data/raw/{source}_{YYYYMMDD}.csv 로 저장."""
    if df is None or df.empty:
        print(f"[SKIP] {source_name}: 저장할 데이터가 없습니다.")
        return None
    ensure_dirs()
    tag = tag or _today_tag()
    path = RAW_DIR / f"{source_name}_{tag}.csv"
    out = df.copy()
    if not isinstance(out.index, pd.RangeIndex):
        out = out.reset_index()
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[SAVE] {path} ({len(df)} rows)")
    return path


def _safe_call(label: str, fn, *args, **kwargs):
    """예외를 삼키고 (결과, 에러메시지) 반환."""
    try:
        return fn(*args, **kwargs), None
    except Exception as exc:
        msg = f"{label} 실패: {exc}"
        print(f"[ERROR] {msg}")
        traceback.print_exc()
        return None, msg


# ---------------------------------------------------------------------------
# 1) FRED
# ---------------------------------------------------------------------------

def fetch_fred_series(
    series_ids: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """
    FRED API로 복수 시리즈를 일별로 수집하고 forward-fill.

    API 키가 없으면 fredgraph CSV 엔드포인트로 폴백한다.
    """
    series_ids = series_ids or list(FRED_SERIES_IDS)
    start_iso, end_iso = _to_iso(start), _to_iso(end)
    frames: list[pd.DataFrame] = []

    for series_id in series_ids:
        try:
            if FRED_API_KEY:
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": series_id,
                    "api_key": FRED_API_KEY,
                    "file_type": "json",
                    "observation_start": start_iso,
                    "observation_end": end_iso,
                }
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                obs = resp.json().get("observations", [])
                rows = [
                    (o["date"], o["value"])
                    for o in obs
                    if o.get("value") not in (None, ".", "")
                ]
                s = pd.DataFrame(rows, columns=["date", series_id])
            else:
                print(f"[WARN] FRED_API_KEY 없음 → CSV 폴백 ({series_id})")
                resp = requests.get(
                    "https://fred.stlouisfed.org/graph/fredgraph.csv",
                    params={"id": series_id},
                    timeout=30,
                )
                resp.raise_for_status()
                s = pd.read_csv(io.StringIO(resp.text))
                s.columns = ["date", series_id]
                s = s[s[series_id].astype(str) != "."]

            s["date"] = pd.to_datetime(s["date"])
            s[series_id] = pd.to_numeric(s[series_id], errors="coerce")
            s = s.dropna(subset=[series_id]).set_index("date").sort_index()
            s = s.loc[pd.Timestamp(start_iso) : pd.Timestamp(end_iso)]
            frames.append(s)
            print(f"[FRED] {series_id}: {len(s)} rows")
            time.sleep(0.2)
        except Exception as exc:
            print(f"[ERROR] FRED {series_id}: {exc}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1, sort=False).sort_index()
    df = df.ffill()
    save_raw_csv(df, "fred")
    return df


# ---------------------------------------------------------------------------
# 2) yfinance
# ---------------------------------------------------------------------------

def fetch_yfinance_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """yfinance로 종가·거래량 수집. MultiIndex를 wide 포맷으로 평탄화."""
    tickers = tickers or list(YFINANCE_TICKERS)
    # yfinance end는 exclusive → +1일
    fetch_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw = yf.download(
        tickers,
        start=_to_iso(start),
        end=fetch_end,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        print("[WARN] yfinance: 빈 응답")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        safe = YFINANCE_COL_PREFIX.get(
            ticker, ticker.replace("^", "").replace(".", "_").replace("-", "_")
        )
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    print(f"[WARN] yfinance 누락: {ticker}")
                    continue
                sub = raw[ticker].copy()
            else:
                # 단일 티커
                sub = raw.copy()

            col_map = {}
            if "Close" in sub.columns:
                col_map["Close"] = f"{safe}_Close"
            if "Volume" in sub.columns:
                col_map["Volume"] = f"{safe}_Volume"
            if not col_map:
                continue
            part = sub[list(col_map.keys())].rename(columns=col_map)
            frames.append(part)
            print(f"[YF] {ticker}: {part.dropna(how='all').shape[0]} rows")
        except Exception as exc:
            print(f"[ERROR] yfinance {ticker}: {exc}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1, sort=False).sort_index()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    save_raw_csv(df, "yfinance")
    return df


# ---------------------------------------------------------------------------
# 3) KRX / LS / FDR 외국인 순매수
# ---------------------------------------------------------------------------

def _load_ls_token_manager():
    """LS증권 폴더의 토큰 매니저를 동적 로드 (프로젝트 config 모듈과 충돌 방지)."""
    import importlib.util
    import os

    if not LS_DIR.exists():
        raise FileNotFoundError(f"LS증권 폴더 없음: {LS_DIR}")

    if LS_APP_KEY:
        os.environ["LS_APP_KEY"] = LS_APP_KEY
    if LS_APP_SECRET_KEY:
        os.environ["LS_APP_SECRET_KEY"] = LS_APP_SECRET_KEY

    if "ls_securities_token" in sys.modules and hasattr(
        sys.modules["ls_securities_token"], "LSSecuritiesTokenManager"
    ):
        tok_mod = sys.modules["ls_securities_token"]
        ls_cfg_mod = sys.modules.get("ls_sec_config")
        if ls_cfg_mod is None:
            raise ImportError("ls_sec_config 모듈이 없습니다. 프로세스를 재시작하세요.")
        return tok_mod.LSSecuritiesTokenManager(), ls_cfg_mod.config

    ls_config_path = LS_DIR / "config.py"
    token_path = LS_DIR / "ls_securities_token.py"
    tok_mod = None
    ls_cfg_mod = None

    try:
        # ls_securities_token 이 `from config import config` 하므로 일시 교체
        spec_cfg = importlib.util.spec_from_file_location("ls_sec_config", ls_config_path)
        if spec_cfg is None or spec_cfg.loader is None:
            raise ImportError(f"LS config 로드 실패: {ls_config_path}")
        ls_cfg_mod = importlib.util.module_from_spec(spec_cfg)
        sys.modules["ls_sec_config"] = ls_cfg_mod
        sys.modules["config"] = ls_cfg_mod
        spec_cfg.loader.exec_module(ls_cfg_mod)

        spec_tok = importlib.util.spec_from_file_location("ls_securities_token", token_path)
        if spec_tok is None or spec_tok.loader is None:
            raise ImportError(f"LS token 모듈 로드 실패: {token_path}")
        tok_mod = importlib.util.module_from_spec(spec_tok)
        sys.modules["ls_securities_token"] = tok_mod
        spec_tok.loader.exec_module(tok_mod)
    finally:
        sys.modules["config"] = project_config

    return tok_mod.LSSecuritiesTokenManager(), ls_cfg_mod.config


def fetch_ls_t1664_foreign_netbuy(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cnt: int = 500,
) -> pd.DataFrame:
    """
    LS t1664 투자자매매종합(차트) — 일별(bdgubun=2), KOSPI(mgubun=1), 금액(vagubun=2).
    외국인 순매수 = tjj17.
    날짜 범위 파라미터가 없어 cnt건 최근 데이터만 반환 → start/end로 필터.
    """
    mgr, ls_config = _load_ls_token_manager()
    headers = mgr.get_auth_header()
    if not headers:
        raise RuntimeError("LS 토큰 발급 실패")

    headers.update(
        {
            "content-type": "application/json; charset=utf-8",
            "tr_cd": "t1664",
            "tr_cont": "N",
            "tr_cont_key": "",
            "mac_address": "",
        }
    )
    url = f"{ls_config.get_base_url()}/stock/investor"
    body = {
        "t1664InBlock": {
            "mgubun": "1",
            "vagubun": "2",
            "bdgubun": "2",
            "cnt": int(cnt),
            "exchgubun": "K",
        }
    }
    # 일부 스펙은 bgubun 키를 사용 → 둘 다 넣어 호환
    body["t1664InBlock"]["bgubun"] = "2"

    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("t1664OutBlock1") or []
    if not rows:
        raise RuntimeError(f"t1664 빈 응답: {data.get('rsp_msg', data)}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["dt"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    df["foreign_netbuy"] = pd.to_numeric(df.get("tjj17"), errors="coerce")
    df["inst_netbuy"] = pd.to_numeric(df.get("tjj18"), errors="coerce")
    df["indiv_netbuy"] = pd.to_numeric(df.get("tjj08"), errors="coerce")
    out = (
        df.dropna(subset=["date"])
        .set_index("date")
        .sort_index()[["foreign_netbuy", "inst_netbuy", "indiv_netbuy"]]
    )
    out = out.loc[pd.Timestamp(_to_iso(start)) : pd.Timestamp(_to_iso(end))]
    save_raw_csv(out, "ls_t1664_foreign")
    return out


def fetch_ls_t1665_foreign_netbuy(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """
    LS t1665 기간별투자자매매추이(차트) — from/to 지원.
    분석 구간(2025-08~)용 주력 소스.

    단위:
      - foreign_netbuy (sa_17 외인계 금액): 백만원
      - foreign_netbuy_qty (sv_17 외인계 수량): 천주
    """
    mgr, ls_config = _load_ls_token_manager()
    headers = mgr.get_auth_header()
    if not headers:
        raise RuntimeError("LS 토큰 발급 실패")

    headers.update(
        {
            "content-type": "application/json; charset=utf-8",
            "tr_cd": "t1665",
            "tr_cont": "N",
            "tr_cont_key": "",
            "mac_address": "",
        }
    )
    url = f"{ls_config.get_base_url()}/stock/chart"
    body = {
        "t1665InBlock": {
            "market": "1",
            "upcode": "001",
            "gubun2": "1",
            "gubun3": "1",
            "from_date": _to_yyyymmdd(start),
            "to_date": _to_yyyymmdd(end),
            "exchgubun": "K",
        }
    }
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("t1665OutBlock1") or []
    if not rows:
        raise RuntimeError(f"t1665 빈 응답: {data.get('rsp_msg', data)}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    # sa_* = 금액(백만원), sv_* = 수량(천주) — LS/Xing 투자자 TR 관례
    df["foreign_netbuy"] = pd.to_numeric(df.get("sa_17"), errors="coerce")
    df["foreign_netbuy_qty"] = pd.to_numeric(df.get("sv_17"), errors="coerce")
    df["inst_netbuy"] = pd.to_numeric(df.get("sa_18"), errors="coerce")
    df["kospi_jisu"] = pd.to_numeric(df.get("jisu"), errors="coerce")
    out = (
        df.dropna(subset=["date"])
        .set_index("date")
        .sort_index()[["foreign_netbuy", "foreign_netbuy_qty", "inst_netbuy", "kospi_jisu"]]
    )
    out.attrs["foreign_netbuy_unit"] = "백만원"
    out.attrs["foreign_netbuy_qty_unit"] = "천주"
    save_raw_csv(out, "ls_t1665_foreign")
    print(
        f"[LS t1665] 외국인 순매수: {len(out)} rows "
        f"(금액=백만원/sa_17, 수량=천주/sv_17)"
    )
    return out


def fetch_ls_vkospi(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    upcode: str = VKOSPI_UPCODE,
) -> pd.DataFrame:
    """
    LS t8429 업종차트(일주월)로 VKOSPI(upcode=205) 일봉 수집.

    OpenAPI 포털 일부 문서의 t8419와 동일 계열이나, 실서버에서는 t8429가 응답한다.
    """
    mgr, ls_config = _load_ls_token_manager()
    headers_base = mgr.get_auth_header()
    if not headers_base:
        raise RuntimeError("LS 토큰 발급 실패")

    url = f"{ls_config.get_base_url()}/indtp/chart"
    rows_all: list[dict[str, Any]] = []
    cts_date = ""
    tr_cont = "N"
    start_s, end_s = _to_yyyymmdd(start), _to_yyyymmdd(end)

    for _ in range(40):
        headers = dict(headers_base)
        headers.update(
            {
                "content-type": "application/json; charset=utf-8",
                "tr_cd": "t8429",
                "tr_cont": tr_cont,
                "tr_cont_key": "",
                "mac_address": "",
            }
        )
        body = {
            "t8429InBlock": {
                "shcode": str(upcode),
                "gubun": "2",  # 2=일봉
                "qrycnt": 500,
                "sdate": start_s,
                "edate": end_s,
                "cts_date": cts_date,
                "comp_yn": "N",
            }
        }
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        block1 = data.get("t8429OutBlock1") or []
        if not block1:
            if not rows_all:
                raise RuntimeError(f"t8429 빈 응답: {data.get('rsp_msg', data)}")
            break
        rows_all.extend(block1)

        cont = (resp.headers.get("tr_cont") or "N").upper()
        next_cts = (data.get("t8429OutBlock") or {}).get("cts_date") or ""
        dates = [str(r.get("date", "")) for r in block1 if r.get("date")]
        if dates and min(dates) <= start_s:
            break
        if cont not in ("Y", "1") or not next_cts:
            break
        cts_date = next_cts
        tr_cont = "Y"
        time.sleep(0.4)

    df = pd.DataFrame(rows_all)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    out = pd.DataFrame(
        {
            "VKOSPI": pd.to_numeric(df["close"], errors="coerce").values,
            "VKOSPI_Open": pd.to_numeric(df.get("open"), errors="coerce").values,
            "VKOSPI_High": pd.to_numeric(df.get("high"), errors="coerce").values,
            "VKOSPI_Low": pd.to_numeric(df.get("low"), errors="coerce").values,
        },
        index=df["date"],
    )
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.loc[pd.Timestamp(_to_iso(start)) : pd.Timestamp(_to_iso(end))]
    out.index.name = "date"
    save_raw_csv(out, "ls_t8429_vkospi")
    print(f"[LS t8429] VKOSPI({upcode}): {len(out)} rows")
    return out


def fetch_ls_index_prices(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """LS t1514로 KOSPI(001)/KOSDAQ(301) 일봉 지수 수집."""
    mgr, ls_config = _load_ls_token_manager()
    headers_base = mgr.get_auth_header()
    if not headers_base:
        raise RuntimeError("LS 토큰 발급 실패")

    url = f"{ls_config.get_base_url()}/indtp/market-data"
    markets = [("001", "KOSPI"), ("301", "KOSDAQ")]
    frames: list[pd.DataFrame] = []

    for upcode, label in markets:
        rows_all: list[dict[str, Any]] = []
        cts_date = " "
        tr_cont = "N"
        for _ in range(30):
            headers = dict(headers_base)
            headers.update(
                {
                    "content-type": "application/json; charset=utf-8",
                    "tr_cd": "t1514",
                    "tr_cont": tr_cont,
                    "tr_cont_key": "",
                    "mac_address": "",
                }
            )
            body = {
                "t1514InBlock": {
                    "upcode": upcode,
                    "gubun1": " ",
                    "gubun2": "1",
                    "cts_date": cts_date,
                    "cnt": 200,
                    "rate_gbn": "1",
                }
            }
            resp = requests.post(url, headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            block1 = data.get("t1514OutBlock1") or []
            rows_all.extend(block1)
            cont = (resp.headers.get("tr_cont") or data.get("tr_cont") or "N").upper()
            next_cts = (data.get("t1514OutBlock") or {}).get("cts_date") or " "
            if cont not in ("Y", "1") or not block1:
                break
            # 이미 start 이전으로 내려갔으면 중단
            dates = [r.get("date") or r.get("dts") for r in block1 if r]
            if dates:
                mind = min(str(d) for d in dates if d)
                if mind and mind < _to_yyyymmdd(start):
                    break
            cts_date = next_cts
            tr_cont = "Y"
            time.sleep(0.4)

        if not rows_all:
            print(f"[WARN] t1514 {label}: 데이터 없음")
            continue

        df = pd.DataFrame(rows_all)
        date_col = "date" if "date" in df.columns else "dts"
        df["date"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
        close_col = "jisu" if "jisu" in df.columns else "close"
        vol_col = "volume" if "volume" in df.columns else None
        part = pd.DataFrame(index=df["date"])
        part[f"LS_{label}_Close"] = pd.to_numeric(df[close_col], errors="coerce").values
        if vol_col and vol_col in df.columns:
            part[f"LS_{label}_Volume"] = pd.to_numeric(df[vol_col], errors="coerce").values
        part = part[~part.index.isna()].sort_index()
        part = part[~part.index.duplicated(keep="last")]
        part = part.loc[pd.Timestamp(_to_iso(start)) : pd.Timestamp(_to_iso(end))]
        frames.append(part)
        print(f"[LS t1514] {label}: {len(part)} rows")
        time.sleep(0.4)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1, sort=False).sort_index()
    out.index.name = "date"
    save_raw_csv(out, "ls_t1514_index")
    return out


def _load_manual_foreign_csv() -> pd.DataFrame:
    """data/manual/ 아래 외국인 순매수 CSV 로드 (컬럼: date, foreign_netbuy)."""
    candidates = sorted(MANUAL_DIR.glob("*foreign*.csv")) + sorted(
        MANUAL_DIR.glob("*외국인*.csv")
    )
    if not candidates:
        return pd.DataFrame()
    path = candidates[-1]
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "일자", "날짜")), None)
    val_col = next(
        (
            c
            for c in df.columns
            if c.lower() in ("foreign_netbuy", "순매수", "외국인순매수", "netbuy")
        ),
        None,
    )
    if date_col is None or val_col is None:
        raise ValueError(f"수동 CSV 컬럼 확인 필요: {path} columns={list(df.columns)}")
    out = pd.DataFrame(
        {
            "foreign_netbuy": pd.to_numeric(df[val_col], errors="coerce").values,
        },
        index=pd.to_datetime(df[date_col]),
    )
    out.index.name = "date"
    out = out.dropna(how="all").sort_index()
    print(f"[MANUAL] 외국인 순매수 로드: {path} ({len(out)} rows)")
    return out


def _fetch_fdr_or_pykrx_foreign(start: str, end: str) -> pd.DataFrame:
    """FinanceDataReader / pykrx 폴백."""
    start_s, end_s = _to_iso(start), _to_iso(end)

    # 1) pykrx
    try:
        from pykrx import stock

        df = stock.get_market_trading_value_by_date(
            _to_yyyymmdd(start), _to_yyyymmdd(end), "KOSPI"
        )
        if df is not None and not df.empty:
            foreign_col = next((c for c in df.columns if "외국인" in str(c)), None)
            if foreign_col is None and "외국인합계" in df.columns:
                foreign_col = "외국인합계"
            if foreign_col is not None:
                out = pd.DataFrame({"foreign_netbuy": pd.to_numeric(df[foreign_col], errors="coerce")})
                out.index = pd.to_datetime(out.index)
                out.index.name = "date"
                print(f"[pykrx] 외국인 순매수: {len(out)} rows")
                return out.sort_index()
    except Exception as exc:
        print(f"[WARN] pykrx 외국인 순매수 실패: {exc}")

    # 2) FinanceDataReader
    try:
        import FinanceDataReader as fdr

        for code in ("KS11", "KRX"):
            try:
                # FDR는 지수/수급 코드가 버전별로 다를 수 있음
                df = fdr.DataReader(code, start_s, end_s)
                if df is not None and not df.empty and "Foreign" in df.columns:
                    out = pd.DataFrame({"foreign_netbuy": pd.to_numeric(df["Foreign"], errors="coerce")})
                    out.index = pd.to_datetime(out.index)
                    out.index.name = "date"
                    print(f"[FDR] 외국인 순매수({code}): {len(out)} rows")
                    return out.sort_index()
            except Exception:
                continue
    except Exception as exc:
        print(f"[WARN] FinanceDataReader 실패: {exc}")

    return pd.DataFrame()


def fetch_krx_foreign_netbuy(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> pd.DataFrame:
    """
    KOSPI 외국인 순매수 수집.

    우선순위:
      1) LS t1665 (기간 지정 가능)
      2) LS t1664 (최근 cnt건 차트)
      3) pykrx / FinanceDataReader
      4) data/manual/*foreign*.csv
      5) KRX OpenAPI skeleton (JSESSIONID 등 사용자 입력)

    TODO: data.krx.co.kr / openapi.krx.co.kr 직접 파싱은 세션·캡차 이슈로
          안정성이 낮아 skeleton만 유지. KRX_JSESSIONID / KRX_AUTH_TOKEN을
          .env에 채우면 아래 _try_krx_openapi 경로를 시도한다.
    """
    # --- LS t1665 ---
    df, err = _safe_call("LS t1665", fetch_ls_t1665_foreign_netbuy, start, end)
    if df is not None and not df.empty:
        save_raw_csv(df, "krx_foreign_netbuy")
        return df

    # --- LS t1664 ---
    df, err = _safe_call("LS t1664", fetch_ls_t1664_foreign_netbuy, start, end)
    if df is not None and not df.empty:
        save_raw_csv(df, "krx_foreign_netbuy")
        return df

    # --- FDR / pykrx ---
    df, err = _safe_call("FDR/pykrx", _fetch_fdr_or_pykrx_foreign, start, end)
    if df is not None and not df.empty:
        save_raw_csv(df, "krx_foreign_netbuy")
        return df

    # --- 수동 CSV ---
    df, err = _safe_call("manual CSV", _load_manual_foreign_csv)
    if df is not None and not df.empty:
        df = df.loc[pd.Timestamp(_to_iso(start)) : pd.Timestamp(_to_iso(end))]
        save_raw_csv(df, "krx_foreign_netbuy")
        return df

    # --- KRX OpenAPI skeleton ---
    df, err = _safe_call("KRX OpenAPI", _try_krx_openapi, start, end)
    if df is not None and not df.empty:
        save_raw_csv(df, "krx_foreign_netbuy")
        return df

    print(
        "[TODO] 외국인 순매수 수집 실패. "
        "data/manual/foreign_netbuy.csv (date, foreign_netbuy) 를 업로드하거나 "
        ".env에 KRX_JSESSIONID를 입력하세요."
    )
    return pd.DataFrame()


def _try_krx_openapi(start: str, end: str) -> pd.DataFrame:
    """
    KRX OpenAPI skeleton.

    TODO:
      - https://openapi.krx.co.kr 로그인 후 발급되는 JSESSIONID / 인증 토큰을
        .env의 KRX_JSESSIONID, KRX_AUTH_TOKEN에 넣으면 호출을 시도한다.
      - 실제 endpoint·payload는 OpenAPI 포털 문서에 맞게 수정 필요.
    """
    if not KRX_JSESSIONID and not KRX_AUTH_TOKEN:
        raise RuntimeError(
            "KRX_JSESSIONID / KRX_AUTH_TOKEN 이 비어 있음 "
            "(https://openapi.krx.co.kr/contents/OPP/MYPG/mypage/OPPMYPG002.cmd)"
        )

    # TODO: 실제 KRX OpenAPI endpoint로 교체
    url = "https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://data.krx.co.kr/",
    }
    cookies = {}
    if KRX_JSESSIONID:
        cookies["JSESSIONID"] = KRX_JSESSIONID
    if KRX_AUTH_TOKEN:
        headers["Authorization"] = KRX_AUTH_TOKEN

    payload = {
        "bld": "dbms/MDC/STAT/standard/MDCSTAT02201",
        "locale": "ko_KR",
        "mktId": "STK",
        "strtDd": _to_yyyymmdd(start),
        "endDd": _to_yyyymmdd(end),
        "share": "1",
        "money": "1",
        "csvxls_isNo": "false",
    }
    resp = requests.post(url, headers=headers, cookies=cookies, data=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = data.get("output") or data.get("OutBlock_1") or []
    if not rows:
        raise RuntimeError(f"KRX 응답에 데이터 없음: keys={list(data.keys())}")

    df = pd.DataFrame(rows)
    # 컬럼명은 KRX 응답에 따라 달라질 수 있음 → 휴리스틱
    date_col = next((c for c in df.columns if "일자" in c or c.lower() == "trd_dd"), df.columns[0])
    foreign_col = next((c for c in df.columns if "외국인" in c and "순매수" in c), None)
    if foreign_col is None:
        raise RuntimeError(f"외국인 순매수 컬럼을 찾지 못함: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "foreign_netbuy": pd.to_numeric(
                df[foreign_col].astype(str).str.replace(",", ""), errors="coerce"
            )
        },
        index=pd.to_datetime(df[date_col]),
    )
    out.index.name = "date"
    return out.sort_index()


# ---------------------------------------------------------------------------
# 4) ECOS (한국은행 기준금리)
# ---------------------------------------------------------------------------

def fetch_ecos_rate(
    stat_code: str = ECOS_BASE_RATE_STAT,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    item_code: str = ECOS_BASE_RATE_ITEM,
) -> pd.DataFrame:
    """한국은행 ECOS API로 기준금리(일별) 수집."""
    if not ECOS_API_KEY:
        raise RuntimeError(
            "ECOS_API_KEY 가 .env에 없습니다. "
            "https://ecos.bok.or.kr/api/ 에서 발급 후 .env에 넣어주세요."
        )

    start_s, end_s = _to_yyyymmdd(start), _to_yyyymmdd(end)
    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{ECOS_API_KEY}"
        f"/json/kr/1/10000/{stat_code}/D/{start_s}/{end_s}/{item_code}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if "StatisticSearch" not in payload:
        # 에러 응답
        raise RuntimeError(f"ECOS 오류 응답: {payload}")

    rows = payload["StatisticSearch"].get("row") or []
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["TIME"], format="%Y%m%d", errors="coerce")
    df["bok_base_rate"] = pd.to_numeric(df["DATA_VALUE"], errors="coerce")
    out = df.dropna(subset=["date"]).set_index("date").sort_index()[["bok_base_rate"]]
    out = out.ffill()
    save_raw_csv(out, "ecos_base_rate")
    print(f"[ECOS] 기준금리: {len(out)} rows")
    return out


# ---------------------------------------------------------------------------
# 오케스트레이션
# ---------------------------------------------------------------------------

def collect_all(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> dict[str, pd.DataFrame]:
    """모든 소스 수집. 개별 실패는 무시하고 성공분만 반환."""
    ensure_dirs()
    print(f"\n===== 데이터 수집 시작: {start} ~ {end} =====\n")
    results: dict[str, pd.DataFrame] = {}

    jobs = [
        ("fred", lambda: fetch_fred_series(FRED_SERIES_IDS, start, end)),
        ("yfinance", lambda: fetch_yfinance_prices(YFINANCE_TICKERS, start, end)),
        ("ls_index", lambda: fetch_ls_index_prices(start, end)),
        ("vkospi", lambda: fetch_ls_vkospi(start, end)),
        ("foreign_netbuy", lambda: fetch_krx_foreign_netbuy(start, end)),
        ("ecos", lambda: fetch_ecos_rate(ECOS_BASE_RATE_STAT, start, end)),
    ]

    for name, fn in jobs:
        print(f"\n--- {name} ---")
        df, err = _safe_call(name, fn)
        if df is not None and not df.empty:
            results[name] = df
        else:
            results[name] = pd.DataFrame()
            if err:
                print(f"[WARN] {name}: 빈 결과 ({err})")

    print("\n===== 수집 요약 =====")
    for name, df in results.items():
        if df is None or df.empty:
            print(f"  {name}: FAIL / empty")
        else:
            print(
                f"  {name}: OK | {df.index.min().date()} ~ {df.index.max().date()} | "
                f"{len(df)} rows | cols={list(df.columns)}"
            )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="유동성 가설용 raw 데이터 수집")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    args = parser.parse_args()
    collect_all(args.start, args.end)

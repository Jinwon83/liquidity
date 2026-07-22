"""
전처리 & 유동성 지표 계산.

- 영업일 정렬 + 주간(금요일) 리샘플
- Fed 순유동성, SOFR-IORB 스프레드
- 지수 2025-08-01=100 리베이스
- 외국인 순매수 누적
- 거래소 휴장(flat) 플래그 — 롤링 상관 전 필터용
- data/processed/master_dataset.csv (outer join)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DEFAULT_END,
    DEFAULT_START,
    PROCESSED_DIR,
    RAW_DIR,
    ensure_dirs,
)
from data_collector import collect_all

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

REBASE_DATE = "2025-08-01"
INDEX_CLOSE_CANDIDATES = [
    "KS11_Close",
    "KQ11_Close",
    "IXIC_Close",
    "GSPC_Close",
    "VIX_Close",
    "DXY_Close",
    "EEM_Close",
    "000660_KS_Close",
    "7709_HK_Close",
    "LS_KOSPI_Close",
    "LS_KOSDAQ_Close",
    "VKOSPI",
]

# FRED 주간/저빈도 + 금리류만 forward-fill (거래소 가격은 ffill 금지)
FFILL_ALLOWED_PREFIXES = (
    "WALCL",
    "RRPONTSYD",
    "WTREGEN",
    "SOFR",
    "IORB",
    "DGS10",
    "DTWEXBGS",
    "BAMLH0A0HYM2",
    "fed_net_liquidity",
    "sofr_iorb_spread",
    "bok_base_rate",
)

# 한국 시장 가격/수급 — 휴장일 NaN 유지
KR_SERIES_PREFIXES = (
    "KS11_",
    "KQ11_",
    "LS_KOSPI",
    "LS_KOSDAQ",
    "000660_KS_",
    "VKOSPI",
    "foreign_netbuy",
    "inst_netbuy",
    "indiv_netbuy",
    "kospi_jisu",
    "KOSPI",
)

# 미국/글로벌 거래소 가격 — 휴장일 NaN 유지
US_SERIES_PREFIXES = (
    "IXIC_",
    "GSPC_",
    "VIX_",
    "DXY_",
    "NASDAQ",
    "VIX",
    "DXY",
)


def _latest_raw(prefix: str) -> Path | None:
    files = sorted(RAW_DIR.glob(f"{prefix}_*.csv"))
    return files[-1] if files else None


def _read_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if c.lower() in ("date", "index")), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()


def load_raw_frames(collected: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """메모리 결과 우선, 없으면 raw CSV에서 로드."""
    mapping = {
        "fred": "fred",
        "yfinance": "yfinance",
        "ls_index": "ls_t1514_index",
        "vkospi": "ls_t8429_vkospi",
        "foreign_netbuy": "krx_foreign_netbuy",
        "ecos": "ecos_base_rate",
    }
    foreign_alts = ["krx_foreign_netbuy", "ls_t1665_foreign", "ls_t1664_foreign"]

    frames: dict[str, pd.DataFrame] = {}
    collected = collected or {}

    for key, prefix in mapping.items():
        if key in collected and collected[key] is not None and not collected[key].empty:
            frames[key] = collected[key].copy()
            frames[key].index = pd.to_datetime(frames[key].index)
            continue

        prefixes = foreign_alts if key == "foreign_netbuy" else [prefix]
        loaded = None
        for p in prefixes:
            path = _latest_raw(p)
            if path is None:
                continue
            try:
                loaded = _read_raw_csv(path)
                print(f"[LOAD] {key} ← {path.name}")
                break
            except Exception as exc:
                print(f"[WARN] {path} 로드 실패: {exc}")
        frames[key] = loaded if loaded is not None else pd.DataFrame()

    return frames


def _is_ffill_col(col: str) -> bool:
    return any(col == p or col.startswith(p) for p in FFILL_ALLOWED_PREFIXES)


def _to_business_daily(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Mon-Fri 캘린더로 정렬.
    - 매크로(FRED 등): forward-fill
    - 거래소 가격/수급: ffill 하지 않음 (휴장 = NaN)
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    bdays = pd.bdate_range(start=start, end=end)
    out = out.reindex(bdays)

    ffill_cols = [c for c in out.columns if _is_ffill_col(c)]
    if ffill_cols:
        out[ffill_cols] = out[ffill_cols].ffill()

    out.index.name = "date"
    return out


def add_liquidity_features(fred: pd.DataFrame) -> pd.DataFrame:
    df = fred.copy()
    for col in ("WALCL", "RRPONTSYD", "WTREGEN", "SOFR", "IORB"):
        if col not in df.columns:
            df[col] = np.nan

    # 단위 정렬: WALCL·WTREGEN = 백만 USD, RRPONTSYD = 십억 USD → *1000
    rrp_mn = df["RRPONTSYD"] * 1000.0
    df["fed_net_liquidity"] = df["WALCL"] - rrp_mn - df["WTREGEN"]
    df["sofr_iorb_spread"] = df["SOFR"] - df["IORB"]
    return df


def rebase_series(series: pd.Series, base_date: str = REBASE_DATE) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    base_ts = pd.Timestamp(base_date)
    if base_ts in s.index and pd.notna(s.loc[base_ts]) and s.loc[base_ts] != 0:
        base_val = float(s.loc[base_ts])
    else:
        after = s.loc[s.index >= base_ts].dropna()
        if after.empty:
            return pd.Series(np.nan, index=s.index)
        base_val = float(after.iloc[0])
    return s / base_val * 100.0


def add_rebased_indexes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in INDEX_CLOSE_CANDIDATES:
        if col in out.columns:
            out[f"{col}_rebased"] = rebase_series(out[col])

    # 표준 별칭 — DXY는 ICE DX-Y.NYB만 사용 (DTWEXBGS와 분리)
    alias = {
        "KS11_Close": "KOSPI",
        "IXIC_Close": "NASDAQ",
        "VIX_Close": "VIX",
        "DXY_Close": "DXY",
        "EEM_Close": "EEM",
    }
    for src, name in alias.items():
        if src in out.columns:
            out[name] = out[src]
            if f"{src}_rebased" in out.columns:
                out[f"{name}_rebased"] = out[f"{src}_rebased"]
        elif name == "KOSPI" and "LS_KOSPI_Close" in out.columns:
            out["KOSPI"] = out["LS_KOSPI_Close"]
            out["KOSPI_rebased"] = rebase_series(out["LS_KOSPI_Close"])

    if "VKOSPI" in out.columns and "VKOSPI_rebased" not in out.columns:
        out["VKOSPI_rebased"] = rebase_series(out["VKOSPI"])
    return out


def add_foreign_cumsum(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "foreign_netbuy" not in out.columns:
        out["foreign_netbuy"] = np.nan
        out["foreign_netbuy_cumsum"] = np.nan
        return out
    # 단위: 백만원 (LS t1665 sa_17). 휴장 NaN은 누적에 0으로 반영.
    nb = pd.to_numeric(out["foreign_netbuy"], errors="coerce")
    out["foreign_netbuy_cumsum"] = nb.fillna(0).cumsum()
    return out


def add_session_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    거래세션·flat(가격 동일 이월) 플래그.

    - is_kr_trading_day: 외국인 수급 또는 LS/YF 한국 지수 관측일이 존재
    - is_us_trading_day: NASDAQ/VIX 등 관측일
    - is_kr_flat / is_us_flat: 직전 유효 종가와 동일 (휴장 ffill 잔재 또는 진짜 보합)
    """
    out = df.copy()

    kr_obs = pd.Series(False, index=out.index)
    for col in ("foreign_netbuy", "LS_KOSPI_Close", "KS11_Close", "VKOSPI"):
        if col in out.columns:
            kr_obs = kr_obs | out[col].notna()
    out["is_kr_trading_day"] = kr_obs

    us_obs = pd.Series(False, index=out.index)
    for col in ("IXIC_Close", "NASDAQ", "VIX_Close", "VIX", "DXY_Close", "DXY"):
        if col in out.columns:
            us_obs = us_obs | out[col].notna()
    out["is_us_trading_day"] = us_obs

    def _flat_mask(price_col: str) -> pd.Series:
        if price_col not in out.columns:
            return pd.Series(False, index=out.index)
        s = pd.to_numeric(out[price_col], errors="coerce")
        return s.notna() & s.eq(s.shift(1))

    out["is_kr_flat"] = _flat_mask("KOSPI") | _flat_mask("KS11_Close")
    out["is_us_flat"] = _flat_mask("NASDAQ") | _flat_mask("IXIC_Close")
    return out


def returns_for_correlation(
    df: pd.DataFrame,
    cols: list[str],
    drop_flat: bool = True,
    require_kr: bool = False,
    require_us: bool = False,
) -> pd.DataFrame:
    """
    롤링/피어슨 상관용 일별 수익률.

    drop_flat=True 이면 가격 flat 구간의 수익률을 NaN 처리해
    휴장 이월(0% 수익률)이 상관을 왜곡하지 않게 한다.
    """
    work = df.copy()
    out = pd.DataFrame(index=work.index)
    for col in cols:
        if col not in work.columns:
            continue
        px = pd.to_numeric(work[col], errors="coerce")
        ret = px.pct_change(fill_method=None)
        if drop_flat:
            flat = px.notna() & px.eq(px.shift(1))
            ret = ret.mask(flat)
        out[col] = ret

    if require_kr and "is_kr_trading_day" in work.columns:
        out = out.where(work["is_kr_trading_day"])
    if require_us and "is_us_trading_day" in work.columns:
        out = out.where(work["is_us_trading_day"])
    return out


def resample_weekly_friday(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.resample("W-FRI").last()


def build_master_dataset(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    collected: dict[str, pd.DataFrame] | None = None,
    collect_if_missing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    master daily + weekly 데이터셋 생성.

    Returns:
        (daily_df, weekly_df)
    """
    ensure_dirs()

    if collect_if_missing and collected is None:
        if not list(RAW_DIR.glob("*.csv")):
            print("[INFO] raw CSV 없음 → collect_all() 실행")
            collected = collect_all(start, end)

    frames = load_raw_frames(collected)

    pieces: list[pd.DataFrame] = []

    if not frames["fred"].empty:
        pieces.append(add_liquidity_features(frames["fred"]))
    if not frames["yfinance"].empty:
        pieces.append(frames["yfinance"])
    if not frames["ls_index"].empty:
        pieces.append(frames["ls_index"])
    if not frames.get("vkospi", pd.DataFrame()).empty:
        pieces.append(frames["vkospi"])
    if not frames["foreign_netbuy"].empty:
        pieces.append(frames["foreign_netbuy"])
    if not frames["ecos"].empty:
        pieces.append(frames["ecos"])

    if not pieces:
        raise RuntimeError("병합할 데이터가 없습니다. 먼저 data_collector를 실행하세요.")

    master = pieces[0]
    for part in pieces[1:]:
        master = master.join(part, how="outer", rsuffix="_dup")
        dup_cols = [c for c in master.columns if c.endswith("_dup")]
        master = master.drop(columns=dup_cols, errors="ignore")

    master = _to_business_daily(master, start, end)
    master = add_rebased_indexes(master)
    master = add_foreign_cumsum(master)
    master = add_session_flags(master)

    # DTWEXBGS는 광의달러지수로 유지. DXY(ICE)와 절대 동일시하지 않음.
    if "DXY" in master.columns and "DTWEXBGS" in master.columns:
        same = (
            master[["DXY", "DTWEXBGS"]]
            .dropna()
            .assign(eq=lambda x: np.isclose(x["DXY"], x["DTWEXBGS"]))
        )
        if not same.empty and same["eq"].all():
            print(
                "[WARN] DXY와 DTWEXBGS가 동일합니다. "
                "DX-Y.NYB 수집이 실패했을 수 있습니다."
            )

    priority = [
        "KOSPI",
        "KOSPI_rebased",
        "NASDAQ",
        "NASDAQ_rebased",
        "fed_net_liquidity",
        "sofr_iorb_spread",
        "DXY",
        "DXY_rebased",
        "EEM",
        "EEM_rebased",
        "DTWEXBGS",
        "foreign_netbuy",
        "foreign_netbuy_cumsum",
        "VIX",
        "VKOSPI",
        "VKOSPI_rebased",
        "bok_base_rate",
        "is_kr_trading_day",
        "is_us_trading_day",
        "is_kr_flat",
        "is_us_flat",
        "WALCL",
        "RRPONTSYD",
        "WTREGEN",
        "SOFR",
        "IORB",
        "DGS10",
        "BAMLH0A0HYM2",
    ]
    ordered = [c for c in priority if c in master.columns]
    rest = [c for c in master.columns if c not in ordered]
    master = master[ordered + rest]

    weekly = resample_weekly_friday(master)

    daily_path = PROCESSED_DIR / "master_dataset.csv"
    weekly_path = PROCESSED_DIR / "master_dataset_weekly.csv"
    master.to_csv(daily_path, encoding="utf-8-sig")
    weekly.to_csv(weekly_path, encoding="utf-8-sig")
    print(f"[SAVE] {daily_path} ({len(master)} rows, {len(master.columns)} cols)")
    print(f"[SAVE] {weekly_path} ({len(weekly)} rows)")

    _print_summary(master)
    return master, weekly


def _print_summary(df: pd.DataFrame) -> None:
    print("\n===== master_dataset 요약 =====")
    print(f"기간: {df.index.min().date()} ~ {df.index.max().date()} ({len(df)} 영업일)")
    print(f"컬럼 수: {len(df.columns)}")
    key_cols = [
        c
        for c in [
            "KOSPI",
            "NASDAQ",
            "fed_net_liquidity",
            "sofr_iorb_spread",
            "DXY",
            "DTWEXBGS",
            "foreign_netbuy",
            "foreign_netbuy_cumsum",
            "VIX",
            "VKOSPI",
            "bok_base_rate",
        ]
        if c in df.columns
    ]
    print("\n핵심 컬럼 결측치:")
    print(df[key_cols].isna().sum().to_string())
    print("foreign_netbuy 단위: 백만원 (LS t1665 sa_17)")
    if "is_kr_trading_day" in df.columns:
        print(
            f"\nKR 거래일: {int(df['is_kr_trading_day'].sum())} / "
            f"US 거래일: {int(df['is_us_trading_day'].sum())} / "
            f"KR flat: {int(df['is_kr_flat'].sum())}"
        )
    if "DXY" in df.columns and "DTWEXBGS" in df.columns:
        both = df[["DXY", "DTWEXBGS"]].dropna()
        if not both.empty:
            print(
                f"DXY 최근={both['DXY'].iloc[-1]:.2f}, "
                f"DTWEXBGS 최근={both['DTWEXBGS'].iloc[-1]:.2f} "
                f"(상관계수={both.corr().iloc[0,1]:.3f})"
            )
    print("\n최근 5영업일:")
    show = [c for c in key_cols if c in df.columns]
    print(df[show].tail().to_string())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="유동성 가설용 feature / master dataset 생성")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--collect",
        action="store_true",
        help="raw 존재 여부와 관계없이 수집부터 다시 실행",
    )
    args = parser.parse_args()

    collected = None
    if args.collect:
        collected = collect_all(args.start, args.end)

    build_master_dataset(
        start=args.start,
        end=args.end,
        collected=collected,
        collect_if_missing=not args.collect,
    )

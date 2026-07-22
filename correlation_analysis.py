"""
KOSPI-NASDAQ 상관관계 분석.

채택 방법론 (원본 월별 표 대조, Method A vs B):
  Method A — 주간 수익률(pct_change)을 YM별 Pearson 상관  ← MAE 0.43 (채택)
  Method B — 일별 수익률 20일 롤링 상관의 월말 값         ← MAE 0.58

Tab 1 롤링 시계열:
  주간 수익률 기준 rolling window (기본 8주 ≈ 월별 표본 규모와 유사).
  차트용으로 주간 시점을 일별 캘린더에 asof forward-fill.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_END, DEFAULT_START, PROCESSED_DIR, ensure_dirs
from feature_builder import returns_for_correlation

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 원본 참조 표 (방법론 검증용)
REFERENCE_MONTHLY = {
    "2025-08": 0.72,
    "2025-09": 0.66,
    "2025-10": 0.95,
    "2025-11": 0.62,
    "2025-12": -0.44,
    "2026-01": -0.86,
    "2026-02": -1.00,
    "2026-03": 0.07,
    "2026-04": 0.77,
    "2026-05": 0.92,
    "2026-06": 0.67,
}

ADOPTED_METHOD = "weekly_return_monthly_pearson"
WEEKLY_ROLL_WINDOW = 8  # Tab1 주간 롤링 윈도우
DAILY_ROLL_WINDOW = 30  # 보조: 일별 30일 롤링 (참고용)


def load_master(kind: str = "daily") -> pd.DataFrame:
    ensure_dirs()
    name = "master_dataset.csv" if kind == "daily" else "master_dataset_weekly.csv"
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. feature_builder를 먼저 실행하세요.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def weekly_returns(weekly: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("KOSPI", "NASDAQ") if c in weekly.columns]
    return weekly[cols].pct_change(fill_method=None)


def monthly_corr_from_weekly_returns(weekly: pd.DataFrame) -> pd.DataFrame:
    """Method A: 주간 수익률 → YM 그룹 Pearson."""
    wr = weekly_returns(weekly)
    wr["YM"] = wr.index.to_period("M").astype(str)
    rows = []
    for ym, g in wr.groupby("YM"):
        pair = g[["KOSPI", "NASDAQ"]].dropna()
        n = len(pair)
        corr = float(pair["KOSPI"].corr(pair["NASDAQ"])) if n >= 2 else np.nan
        low_n = bool(n <= 4)
        label = f"{corr:.3f}*" if (pd.notna(corr) and low_n) else (
            f"{corr:.3f}" if pd.notna(corr) else ""
        )
        rows.append(
            {
                "YM": ym,
                "corr": corr,
                "n_weeks": n,
                "low_n": low_n,
                "corr_label": label,  # n_weeks<=4 이면 '*' 표본부족 경고
            }
        )
    out = pd.DataFrame(rows).set_index("YM").sort_index()
    out["method"] = ADOPTED_METHOD
    return out


def _pair_corr(x: pd.Series, y: pd.Series, min_n: int = 5) -> tuple[float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < min_n:
        return float("nan"), n
    return float(x[mask].corr(y[mask])), n


def calendar_end_dummies(
    index: pd.DatetimeIndex, n_bd: int = 3
) -> tuple[pd.Series, pd.Series]:
    """
    인덱스에 나타난 날짜 기준 캘린더 더미.

    - is_month_end: 매월 마지막 n_bd 영업일(인덱스에 존재하는 일)
    - is_quarter_end: 분기말 월(3·6·9·12)의 마지막 n_bd 영업일
    """
    idx = pd.DatetimeIndex(index)
    is_month_end = pd.Series(False, index=idx, dtype=bool)
    for _, g in pd.Series(1, index=idx).groupby(idx.to_period("M")):
        last = g.index.sort_values()[-n_bd:]
        is_month_end.loc[last] = True

    q_months = {3, 6, 9, 12}
    is_quarter_end = pd.Series(
        is_month_end.to_numpy() & np.isin(idx.month, list(q_months)),
        index=idx,
        dtype=bool,
    )
    return is_month_end, is_quarter_end


def _regime_corr_row(
    label: str,
    mask: pd.Series,
    rrp: pd.Series,
    spread: pd.Series,
    spread_vol20: pd.Series,
    rrp_depleted: pd.Series,
    n_excluded: int = 0,
    note: str = "",
) -> dict:
    corr_lvl, n_lvl = _pair_corr(rrp[mask], spread[mask])
    corr_vol, n_vol = _pair_corr(rrp[mask], spread_vol20[mask])
    rrp_mean = float(rrp[mask].mean()) if mask.any() else np.nan
    rrp_med = float(rrp[mask].median()) if mask.any() else np.nan
    depleted_share = float(rrp_depleted[mask].mean()) if mask.any() else np.nan
    return {
        "regime": label,
        "rrp_mean": rrp_mean,
        "rrp_median": rrp_med,
        "rrp_depleted_share": depleted_share,
        "corr_rrp_vs_spread": corr_lvl,
        "abs_corr_rrp_vs_spread": abs(corr_lvl) if pd.notna(corr_lvl) else np.nan,
        "n_level": n_lvl,
        "corr_rrp_vs_spread_vol20": corr_vol,
        "abs_corr_rrp_vs_vol20": abs(corr_vol) if pd.notna(corr_vol) else np.nan,
        "n_vol": n_vol,
        "n_excluded": n_excluded,
        "note": note,
    }


def analyze_rrp_vs_spread(
    daily: pd.DataFrame,
    window: int = 20,
    rrp_depleted_threshold: float = 10.0,
    regime1_end: str = "2025-10-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    RRPONTSYD·Fed순유동성 vs SOFR-IORB 상관 + 구간(여유/고갈)별 RRP 상관.

    구간1: start ~ regime1_end (기본 2025-08~10, RRP 상대적 여유)
    구간2: regime1_end 다음날 ~ end (11월~, RRP 고갈 국면)
    추가로 구간2에서 월말·분기말(각 마지막 3영업일) 제외 상관을 계산.

    Returns:
        (전체 관계 요약, panel2 시계열, 구간별 RRP-스프레드 비교표)
    """
    df = daily.copy()
    for c in ("RRPONTSYD", "sofr_iorb_spread", "fed_net_liquidity"):
        if c not in df.columns:
            raise KeyError(f"master에 {c} 컬럼 없음")

    spread = pd.to_numeric(df["sofr_iorb_spread"], errors="coerce")
    rrp = pd.to_numeric(df["RRPONTSYD"], errors="coerce")
    fed = pd.to_numeric(df["fed_net_liquidity"], errors="coerce")
    spread_vol20 = spread.rolling(window, min_periods=max(5, window // 2)).std()
    rrp_depleted = rrp < rrp_depleted_threshold
    is_month_end, is_quarter_end = calendar_end_dummies(df.index, n_bd=3)

    # Fed 순유동성: 주간 리샘플 → 일별 ffill (Panel2 좌축)
    fed_weekly = fed.resample("W-FRI").last()
    fed_weekly_ffill = (
        fed_weekly.reindex(fed.index.union(fed_weekly.index)).sort_index().ffill()
    )
    fed_weekly_ffill = fed_weekly_ffill.reindex(fed.index)

    pairs = [
        ("RRPONTSYD vs sofr_iorb_spread (level) [전체]", rrp, spread),
        ("RRPONTSYD vs sofr_iorb_spread_vol20 [전체]", rrp, spread_vol20),
        ("fed_net_liquidity vs sofr_iorb_spread (level) [전체]", fed, spread),
        ("fed_net_liquidity vs sofr_iorb_spread_vol20 [전체]", fed, spread_vol20),
    ]
    rows = []
    for name, x, y in pairs:
        corr, n = _pair_corr(x, y)
        rows.append(
            {
                "relationship": name,
                "corr": corr,
                "abs_corr": abs(corr) if pd.notna(corr) else np.nan,
                "n": n,
            }
        )
    summary = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)

    # 구간별 RRP vs spread
    cut = pd.Timestamp(regime1_end)
    r1 = df.index <= cut
    r2 = df.index > cut
    cal_ex = is_month_end | is_quarter_end  # 월말 ⊃ 분기말, OR로 명시
    r2_ex = r2 & ~cal_ex
    n_ex_r2 = int((r2 & cal_ex).sum())

    regime_rows = [
        _regime_corr_row(
            "구간1 (2025-08~10, RRP 여유)",
            r1,
            rrp,
            spread,
            spread_vol20,
            rrp_depleted,
            note="baseline",
        ),
        _regime_corr_row(
            "구간2 (2025-11~, RRP 고갈)",
            r2,
            rrp,
            spread,
            spread_vol20,
            rrp_depleted,
            note="baseline_incl_month_quarter_end",
        ),
        _regime_corr_row(
            "구간2 (월말·분기말 제외)",
            r2_ex,
            rrp,
            spread,
            spread_vol20,
            rrp_depleted,
            n_excluded=n_ex_r2,
            note="exclude is_month_end|is_quarter_end (last 3 BD)",
        ),
        _regime_corr_row(
            "구간2 (분기말만 제외)",
            r2 & ~is_quarter_end,
            rrp,
            spread,
            spread_vol20,
            rrp_depleted,
            n_excluded=int((r2 & is_quarter_end).sum()),
            note="exclude is_quarter_end only",
        ),
    ]
    regime_df = pd.DataFrame(regime_rows)

    panel2 = pd.DataFrame(
        {
            "RRPONTSYD": rrp,
            "sofr_iorb_spread": spread,
            "sofr_iorb_vol20": spread_vol20,
            "fed_net_liquidity": fed,
            "fed_net_liquidity_weekly": fed_weekly_ffill,
            "rrp_depleted": rrp_depleted.astype(bool),
            "is_month_end": is_month_end.astype(bool),
            "is_quarter_end": is_quarter_end.astype(bool),
        },
        index=df.index,
    )
    return summary, panel2, regime_df


def _pick_regime(regime_df: pd.DataFrame, key: str) -> pd.Series | None:
    hit = regime_df[regime_df["regime"].astype(str).str.contains(key, regex=False)]
    if hit.empty:
        return None
    return hit.iloc[0]


def judge_rrp_regime(regime_df: pd.DataFrame) -> str:
    """구간1 vs 구간2 + 월말·분기말 제외 후 상관으로 최종 판정."""
    row1 = _pick_regime(regime_df, "구간1")
    row2 = _pick_regime(regime_df, "구간2 (2025-11")
    row2x = _pick_regime(regime_df, "월말·분기말 제외")
    if row1 is None or row2 is None:
        return "판정 불가 (구간 데이터 부족)"

    a1 = float(row1["abs_corr_rrp_vs_spread"])
    a2 = float(row2["abs_corr_rrp_vs_spread"])
    c1 = float(row1["corr_rrp_vs_spread"])
    c2 = float(row2["corr_rrp_vs_spread"])
    weak = 0.2
    if pd.isna(a1) or pd.isna(a2):
        return "판정 불가 (결측)"

    if a1 < weak and a2 < weak:
        base = (
            f"애초에 무관에 가깝음 "
            f"(구간1 |corr|={a1:.3f}, 구간2 |corr|={a2:.3f} 모두 <{weak})"
        )
    elif a1 >= weak and a2 < weak:
        base = (
            f"고갈되며 설명력을 잃음 "
            f"(구간1 |corr|={a1:.3f} → 구간2 |corr|={a2:.3f}; "
            f"corr {c1:+.3f} → {c2:+.3f})"
        )
    elif a1 < weak and a2 >= weak:
        base = (
            f"고갈 국면에서 오히려 동행성 증가 "
            f"(구간1 |corr|={a1:.3f} → 구간2 |corr|={a2:.3f})"
        )
    elif c1 * c2 < 0 and min(a1, a2) >= weak:
        base = (
            f"강도는 유지됐으나 부호가 반전 — 고갈 전후 '관계의 성격'이 바뀜 "
            f"(구간1 corr={c1:+.3f} → 구간2 corr={c2:+.3f}; "
            f"|corr| {a1:.3f}≈{a2:.3f}). "
            f"단순 '고갈→설명력 상실'이라기보다 레짐 전환."
        )
    elif a1 > a2 * 1.25:
        base = (
            f"여유 구간에서 더 강했고 고갈 후 약화 "
            f"(구간1 |corr|={a1:.3f} → 구간2 |corr|={a2:.3f})"
        )
    elif a2 > a1 * 1.25:
        base = (
            f"고갈 이후에도(또는 더) 강함 "
            f"(구간1 |corr|={a1:.3f} → 구간2 |corr|={a2:.3f})"
        )
    else:
        base = (
            f"두 구간 모두 유사 강도 "
            f"(구간1 |corr|={a1:.3f}, 구간2 |corr|={a2:.3f}; "
            f"corr {c1:+.3f} → {c2:+.3f})"
        )

    if row2x is None or pd.isna(row2x["corr_rrp_vs_spread"]):
        return base + " | 월말·분기말 제외 상관 계산 불가"

    c2x = float(row2x["corr_rrp_vs_spread"])
    a2x = float(row2x["abs_corr_rrp_vs_spread"])
    n_ex = int(row2x.get("n_excluded", 0) or 0)
    n_keep = int(row2x["n_level"])
    cal = (
        f"구간2 월말·분기말 제외 후 corr={c2x:+.3f} "
        f"(제외 전 {c2:+.3f} → 제외 후 {c2x:+.3f}; "
        f"n={n_keep}, 제외일={n_ex})"
    )

    # 양의 상관이 유의미히 남으면 진짜 레짐, 거의 사라지면 기술적 효과
    remain_threshold = 0.20
    vanish_threshold = 0.15
    if c2 > 0 and c2x > 0 and a2x >= remain_threshold:
        final = (
            f"최종: 진짜 레짐 전환 "
            f"(월말·분기말 제외 후에도 양의 상관 유지, |corr|={a2x:.3f}≥{remain_threshold})"
        )
    elif a2x < vanish_threshold or (c2 > 0 and c2x <= 0) or (
        a2 > 0 and a2x < 0.5 * a2
    ):
        final = (
            f"최종: 분기말·월말 기술적 효과였음 "
            f"(제외 후 |corr|={a2x:.3f}로 거의 소멸/약화; "
            f"제외 전 +{c2:.3f} → 제외 후 {c2x:+.3f})"
        )
    else:
        final = (
            f"최종: 부분적 — 캘린더 효과가 일부이나 잔여 상관도 존재 "
            f"(제외 전 {c2:+.3f} → 제외 후 {c2x:+.3f})"
        )
    return f"{base} | {cal} | {final}"


def analyze_korea_alpha(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """
    외국인 순매수·EEM·DXY·코리아 알파 상관 분석.

    Returns:
        (전체 관계 요약, 월별 3상관 표, panel3 시계열, 최강 조합 판정문)
    """
    df = daily.copy()
    need = ["KOSPI", "EEM", "DXY", "foreign_netbuy", "fed_net_liquidity"]
    # EEM may only exist as EEM_Close
    if "EEM" not in df.columns and "EEM_Close" in df.columns:
        df["EEM"] = df["EEM_Close"]
    if "DXY" not in df.columns and "DXY_Close" in df.columns:
        df["DXY"] = df["DXY_Close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError(f"korea alpha 분석에 필요 컬럼 없음: {missing}")

    kospi_ret = pd.to_numeric(df["KOSPI"], errors="coerce").pct_change(fill_method=None)
    eem_ret = pd.to_numeric(df["EEM"], errors="coerce").pct_change(fill_method=None)
    dxy_chg = pd.to_numeric(df["DXY"], errors="coerce").pct_change(fill_method=None)
    netbuy = pd.to_numeric(df["foreign_netbuy"], errors="coerce")  # 일별 원본 (누적 X)
    fed = pd.to_numeric(df["fed_net_liquidity"], errors="coerce")
    alpha = kospi_ret - eem_ret  # 코리아 알파

    pairs = [
        ("foreign_netbuy vs DXY_pct_change", netbuy, dxy_chg),
        ("foreign_netbuy vs EEM_ret", netbuy, eem_ret),
        ("korea_alpha vs DXY_pct_change", alpha, dxy_chg),
        ("korea_alpha vs fed_net_liquidity", alpha, fed),
        ("KOSPI_ret vs EEM_ret (전체)", kospi_ret, eem_ret),
        ("KOSPI_ret vs DXY_pct_change (전체)", kospi_ret, dxy_chg),
    ]
    rows = []
    for name, x, y in pairs:
        corr, n = _pair_corr(x, y)
        rows.append(
            {
                "relationship": name,
                "corr": corr,
                "abs_corr": abs(corr) if pd.notna(corr) else np.nan,
                "n": n,
            }
        )
    summary = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)

    # 월별 3상관
    frame = pd.DataFrame(
        {
            "kospi_ret": kospi_ret,
            "eem_ret": eem_ret,
            "dxy_chg": dxy_chg,
            "alpha": alpha,
            "fed": fed,
            "foreign_netbuy": netbuy,
        },
        index=df.index,
    )
    frame["YM"] = frame.index.to_period("M").astype(str)
    monthly_rows = []
    for ym, g in frame.groupby("YM"):
        c_em, n_em = _pair_corr(g["kospi_ret"], g["eem_ret"])
        c_dxy, n_dxy = _pair_corr(g["kospi_ret"], g["dxy_chg"])
        c_al, n_al = _pair_corr(g["alpha"], g["fed"])
        monthly_rows.append(
            {
                "YM": ym,
                "KOSPI_ret_vs_EEM_ret": c_em,
                "n_em": n_em,
                "KOSPI_ret_vs_DXY_chg": c_dxy,
                "n_dxy": n_dxy,
                "alpha_vs_fed_net_liquidity": c_al,
                "n_alpha": n_al,
            }
        )
    monthly = pd.DataFrame(monthly_rows).set_index("YM").sort_index()

    # Panel3: 리베이스 지수 + 일별 순매수
    kospi_reb = df["KOSPI_rebased"] if "KOSPI_rebased" in df.columns else rebase_series_safe(df["KOSPI"])
    if "EEM_rebased" in df.columns:
        eem_reb = df["EEM_rebased"]
    else:
        eem_reb = rebase_series_safe(df["EEM"])

    panel3 = pd.DataFrame(
        {
            "KOSPI_rebased": kospi_reb,
            "EEM_rebased": eem_reb,
            "foreign_netbuy": netbuy,
            "kospi_ret": kospi_ret,
            "eem_ret": eem_ret,
            "korea_alpha": alpha,
            "dxy_chg": dxy_chg,
            "fed_net_liquidity": fed,
        },
        index=df.index,
    )

    best = summary.iloc[0]
    verdict = (
        f"가장 설명력 큰 조합: {best['relationship']} "
        f"(corr={best['corr']:+.3f}, |corr|={best['abs_corr']:.3f}, n={int(best['n'])})"
    )
    # 월별 평균 |corr|로도 보조 판정
    m_abs = {
        "KOSPI↔EEM (커플링)": float(monthly["KOSPI_ret_vs_EEM_ret"].abs().mean()),
        "KOSPI↔DXY (달러 직결)": float(monthly["KOSPI_ret_vs_DXY_chg"].abs().mean()),
        "alpha↔Fed순유동성 (한국 고유분)": float(
            monthly["alpha_vs_fed_net_liquidity"].abs().mean()
        ),
    }
    best_m = max(m_abs, key=m_abs.get)
    verdict += (
        f" | 월별 평균|corr| 기준 최강: {best_m} "
        + ", ".join(f"{k}={v:.3f}" for k, v in m_abs.items())
    )
    return summary, monthly, panel3, verdict


def rebase_series_safe(s: pd.Series, base_date: str = "2025-08-01") -> pd.Series:
    from feature_builder import rebase_series

    return rebase_series(pd.to_numeric(s, errors="coerce"), base_date=base_date)


def build_panel1_background(daily: pd.DataFrame, weekly: pd.DataFrame) -> pd.Series:
    """Fed 순유동성 주간 리샘플 → 일별 asof ffill (Panel1 배경)."""
    if "fed_net_liquidity" in weekly.columns:
        w = pd.to_numeric(weekly["fed_net_liquidity"], errors="coerce")
    else:
        w = (
            pd.to_numeric(daily["fed_net_liquidity"], errors="coerce")
            .resample("W-FRI")
            .last()
        )
    return expand_weekly_to_daily(w, daily.index)


def month_end_rolling20_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Method B (비교용): 일별 수익률 20일 롤링 → 월 마지막 유효값."""
    ret = returns_for_correlation(daily, ["KOSPI", "NASDAQ"], drop_flat=True)
    roll = ret["KOSPI"].rolling(20, min_periods=10).corr(ret["NASDAQ"])
    rows = []
    for ym, g in roll.groupby(roll.index.to_period("M")):
        valid = g.dropna()
        if valid.empty:
            rows.append({"YM": str(ym), "corr": np.nan, "asof": None})
        else:
            rows.append(
                {
                    "YM": str(ym),
                    "corr": float(valid.iloc[-1]),
                    "asof": valid.index[-1].strftime("%Y-%m-%d"),
                }
            )
    return pd.DataFrame(rows).set_index("YM").sort_index()


def compare_to_reference(monthly_a: pd.DataFrame, monthly_b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ym, target in REFERENCE_MONTHLY.items():
        a = monthly_a["corr"].get(ym, np.nan)
        b = monthly_b["corr"].get(ym, np.nan)
        rows.append(
            {
                "YM": ym,
                "orig": target,
                "method_A_weekly": a,
                "diff_A": a - target if pd.notna(a) else np.nan,
                "method_B_roll20me": b,
                "diff_B": b - target if pd.notna(b) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rolling_corr_weekly(
    weekly: pd.DataFrame,
    window: int = WEEKLY_ROLL_WINDOW,
    min_periods: int | None = None,
) -> pd.Series:
    """주간 수익률 롤링 상관 (채택 방법론의 시계열 확장)."""
    min_periods = min_periods or max(3, window // 2)
    wr = weekly_returns(weekly)
    return wr["KOSPI"].rolling(window, min_periods=min_periods).corr(wr["NASDAQ"])


def rolling_corr_daily(
    daily: pd.DataFrame,
    window: int = DAILY_ROLL_WINDOW,
    min_periods: int | None = None,
) -> pd.Series:
    """일별 수익률 롤링 상관 (참고용). flat 구간 제외."""
    min_periods = min_periods or max(10, window // 2)
    ret = returns_for_correlation(daily, ["KOSPI", "NASDAQ"], drop_flat=True)
    return ret["KOSPI"].rolling(window, min_periods=min_periods).corr(ret["NASDAQ"])


def expand_weekly_to_daily(weekly_series: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """주간 롤링값을 일별 인덱스에 asof forward-fill."""
    s = weekly_series.dropna().sort_index()
    if s.empty:
        return pd.Series(np.nan, index=daily_index)
    out = s.reindex(daily_index.union(s.index)).sort_index().ffill()
    return out.reindex(daily_index)


def run_analysis(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> dict[str, pd.DataFrame | pd.Series]:
    ensure_dirs()
    daily = load_master("daily").loc[start:end]
    weekly = load_master("weekly").loc[start:end]

    monthly_a = monthly_corr_from_weekly_returns(weekly)
    monthly_b = month_end_rolling20_from_daily(daily)
    comparison = compare_to_reference(monthly_a, monthly_b)

    mae_a = comparison["diff_A"].abs().mean()
    mae_b = comparison["diff_B"].abs().mean()
    adopted = "A" if mae_a <= mae_b else "B"

    print("=" * 72)
    print("METHOD A vs B vs 원본 표")
    print("=" * 72)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print(f"\nMAE_A={mae_a:.4f}  MAE_B={mae_b:.4f}  → 채택: Method {adopted}")
    print(f"채택 정의: {ADOPTED_METHOD}")

    # 채택 방법론 기반 산출물
    roll_weekly = rolling_corr_weekly(weekly, WEEKLY_ROLL_WINDOW)
    roll_weekly_daily = expand_weekly_to_daily(roll_weekly, daily.index)
    roll_daily_30 = rolling_corr_daily(daily, DAILY_ROLL_WINDOW)
    fed_weekly_bg = build_panel1_background(daily, weekly)

    # RRP vs 스프레드 관계 (+ 구간1/2)
    rrp_summary, panel2, regime_df = analyze_rrp_vs_spread(daily, window=20)
    regime_verdict = judge_rrp_regime(regime_df)

    # 코리아 알파 / EEM / 외국인 순매수
    alpha_summary, alpha_monthly, panel3, alpha_verdict = analyze_korea_alpha(daily)

    # 저장
    comparison_path = PROCESSED_DIR / "corr_method_A_vs_B.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    monthly_path = PROCESSED_DIR / "corr_monthly_adopted.csv"
    monthly_a.to_csv(monthly_path, encoding="utf-8-sig")

    # Tab1용 메인 시계열 + Fed 순유동성(주간) 배경
    tab1 = pd.DataFrame(
        {
            "corr_weekly_roll8_ffill": roll_weekly_daily,
            "corr_daily_roll30": roll_daily_30,
            "fed_net_liquidity_weekly": fed_weekly_bg,
            "KOSPI": daily.get("KOSPI"),
            "NASDAQ": daily.get("NASDAQ"),
            "KOSPI_rebased": daily.get("KOSPI_rebased"),
            "NASDAQ_rebased": daily.get("NASDAQ_rebased"),
        },
        index=daily.index,
    )
    tab1_path = PROCESSED_DIR / "corr_rolling_tab1.csv"
    tab1.to_csv(tab1_path, encoding="utf-8-sig")

    weekly_roll_path = PROCESSED_DIR / "corr_rolling_weekly.csv"
    pd.DataFrame({"corr_weekly_roll8": roll_weekly}).to_csv(
        weekly_roll_path, encoding="utf-8-sig"
    )

    panel2_path = PROCESSED_DIR / "panel2_rrp_sofr.csv"
    panel2.to_csv(panel2_path, encoding="utf-8-sig")

    rrp_path = PROCESSED_DIR / "rrp_spread_corr_summary.csv"
    rrp_summary.to_csv(rrp_path, index=False, encoding="utf-8-sig")

    regime_path = PROCESSED_DIR / "rrp_regime_corr.csv"
    regime_df.to_csv(regime_path, index=False, encoding="utf-8-sig")

    panel3_path = PROCESSED_DIR / "panel3_korea_alpha.csv"
    panel3.to_csv(panel3_path, encoding="utf-8-sig")

    alpha_sum_path = PROCESSED_DIR / "korea_alpha_corr_summary.csv"
    alpha_summary.to_csv(alpha_sum_path, index=False, encoding="utf-8-sig")

    alpha_m_path = PROCESSED_DIR / "korea_alpha_monthly_corr.csv"
    alpha_monthly.to_csv(alpha_m_path, encoding="utf-8-sig")

    meta = {
        "adopted_method": ADOPTED_METHOD,
        "adopted_code": adopted,
        "mae_A": float(mae_a),
        "mae_B": float(mae_b),
        "weekly_roll_window": WEEKLY_ROLL_WINDOW,
        "low_n_threshold": 4,
        "rrp_depleted_threshold_bn": 10.0,
        "rrp_regime_verdict": regime_verdict,
        "korea_alpha_verdict": alpha_verdict,
        "note": (
            "Panel2: Fed순유동성(주간)+SOFR-IORB+rrp_depleted 음영. "
            "Panel3: KOSPI/EEM 리베이스 + 일별 foreign_netbuy. "
            "코리아 알파 = KOSPI_ret - EEM_ret."
        ),
    }
    meta_path = PROCESSED_DIR / "corr_method_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[SAVE] {comparison_path}")
    print(f"[SAVE] {monthly_path}")
    print(f"[SAVE] {tab1_path}")
    print(f"[SAVE] {weekly_roll_path}")
    print(f"[SAVE] {panel2_path}")
    print(f"[SAVE] {rrp_path}")
    print(f"[SAVE] {regime_path}")
    print(f"[SAVE] {panel3_path}")
    print(f"[SAVE] {alpha_sum_path}")
    print(f"[SAVE] {alpha_m_path}")
    print(f"[SAVE] {meta_path}")

    print("\n===== 채택 Method A 월별 상관 (전체, * = n_weeks<=4) =====")
    show = monthly_a.copy()
    print(show[["corr", "n_weeks", "low_n", "corr_label"]].to_string(float_format=lambda x: f"{x: .4f}"))

    print("\n===== RRP / Fed순유동성 vs SOFR-IORB 상관 (전체) =====")
    print(rrp_summary.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

    print("\n===== 구간별 RRPONTSYD vs sofr_iorb_spread =====")
    print(regime_df.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print(f"\n판정: {regime_verdict}")

    print("\n===== 코리아 알파 / EEM / 외국인 순매수 상관 =====")
    print(alpha_summary.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print("\n===== 월별: KOSPI↔EEM | KOSPI↔DXY | alpha↔Fed =====")
    print(alpha_monthly.to_string(float_format=lambda x: f"{x: .4f}"))
    print(f"\n판정: {alpha_verdict}")

    print("\n===== Tab1 주간 롤링(8주) 최근 8개 =====")
    print(roll_weekly.dropna().tail(8).to_string(float_format=lambda x: f"{x: .4f}"))

    return {
        "comparison": comparison,
        "monthly_a": monthly_a,
        "monthly_b": monthly_b,
        "tab1": tab1,
        "roll_weekly": roll_weekly,
        "rrp_summary": rrp_summary,
        "regime_df": regime_df,
        "panel2": panel2,
        "alpha_summary": alpha_summary,
        "alpha_monthly": alpha_monthly,
        "panel3": panel3,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    args = p.parse_args()
    run_analysis(args.start, args.end)

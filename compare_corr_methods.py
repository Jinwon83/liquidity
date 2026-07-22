"""Compare KOSPI-NASDAQ correlation methodologies vs original monthly table."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
ORIG = {
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
YMS = list(ORIG.keys())
TARGETS = np.array([ORIG[y] for y in YMS], dtype=float)


def mae_rmse(vals: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(vals, dtype=float) - TARGETS
    return float(np.nanmean(np.abs(diff))), float(np.sqrt(np.nanmean(diff**2)))


def month_end_of_series(s: pd.Series, ym: str) -> float:
    m = s.loc[s.index.to_period("M").astype(str) == ym].dropna()
    if m.empty:
        return float("nan")
    return float(m.iloc[-1])


def main() -> None:
    weekly = pd.read_csv(
        ROOT / "data/processed/master_dataset_weekly.csv",
        index_col=0,
        parse_dates=True,
    )
    daily = pd.read_csv(
        ROOT / "data/processed/master_dataset.csv",
        index_col=0,
        parse_dates=True,
    )

    # ---- A: weekly returns, monthly corr ----
    wr = weekly[["KOSPI", "NASDAQ"]].pct_change(fill_method=None)
    wr["YM"] = wr.index.to_period("M").astype(str)
    a_vals = []
    a_n = []
    for ym in YMS:
        g = wr.loc[wr["YM"] == ym, ["KOSPI", "NASDAQ"]].dropna()
        a_n.append(len(g))
        a_vals.append(float(g["KOSPI"].corr(g["NASDAQ"])) if len(g) >= 2 else np.nan)
    a_vals = np.array(a_vals, dtype=float)
    mae_a, rmse_a = mae_rmse(a_vals)

    print("=" * 72)
    print("METHOD A: weekly pct_change -> groupby YM -> Pearson corr")
    print("=" * 72)
    A = pd.DataFrame(
        {
            "YM": YMS,
            "orig": TARGETS,
            "method_A": a_vals,
            "n_weeks": a_n,
            "diff_A": a_vals - TARGETS,
        }
    )
    print(A.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print(f"MAE_A={mae_a:.4f}  RMSE_A={rmse_a:.4f}")
    print()

    # ---- B: 20d rolling daily, month-end ----
    ret = daily[["KOSPI", "NASDAQ"]].pct_change(fill_method=None)
    roll20 = ret["KOSPI"].rolling(20, min_periods=10).corr(ret["NASDAQ"])
    b_vals = np.array([month_end_of_series(roll20, ym) for ym in YMS], dtype=float)
    b_asof = []
    for ym in YMS:
        m = roll20.loc[roll20.index.to_period("M").astype(str) == ym].dropna()
        b_asof.append(m.index[-1].date().isoformat() if len(m) else None)
    mae_b, rmse_b = mae_rmse(b_vals)

    print("=" * 72)
    print("METHOD B: daily return 20d rolling corr -> month last trading day")
    print("=" * 72)
    B = pd.DataFrame(
        {
            "YM": YMS,
            "orig": TARGETS,
            "method_B": b_vals,
            "asof": b_asof,
            "diff_B": b_vals - TARGETS,
        }
    )
    print(B.to_string(index=False, float_format=lambda x: f"{x: .4f}"))
    print(f"MAE_B={mae_b:.4f}  RMSE_B={rmse_b:.4f}")
    print()

    # ---- Extra variants to find best match ----
    variants: dict[str, np.ndarray] = {
        "A_weekly_monthly": a_vals,
        "B_roll20_me": b_vals,
    }

    # daily returns monthly corr
    dr = daily[["KOSPI", "NASDAQ"]].pct_change(fill_method=None)
    dr["YM"] = dr.index.to_period("M").astype(str)
    variants["daily_monthly"] = np.array(
        [
            float(
                dr.loc[dr["YM"] == ym, ["KOSPI", "NASDAQ"]]
                .dropna()["KOSPI"]
                .corr(dr.loc[dr["YM"] == ym, ["KOSPI", "NASDAQ"]].dropna()["NASDAQ"])
            )
            if len(dr.loc[dr["YM"] == ym, ["KOSPI", "NASDAQ"]].dropna()) >= 5
            else np.nan
            for ym in YMS
        ]
    )

    roll30 = ret["KOSPI"].rolling(30, min_periods=15).corr(ret["NASDAQ"])
    variants["roll30_me"] = np.array([month_end_of_series(roll30, ym) for ym in YMS])

    # US-lead daily monthly
    dlag = daily[["KOSPI", "NASDAQ"]].copy()
    dlag["NASDAQ"] = dlag["NASDAQ"].shift(1)
    dlr = dlag.pct_change(fill_method=None)
    dlr["YM"] = dlr.index.to_period("M").astype(str)
    variants["daily_uslag1_monthly"] = np.array(
        [
            float(g["KOSPI"].corr(g["NASDAQ"]))
            if len(g := dlr.loc[dlr["YM"] == ym, ["KOSPI", "NASDAQ"]].dropna()) >= 5
            else np.nan
            for ym in YMS
        ]
    )

    # US-lead 20d roll ME
    ret_u = daily[["KOSPI"]].join(daily["NASDAQ"].shift(1).rename("NASDAQ"))
    ret_u = ret_u.pct_change(fill_method=None)
    roll20u = ret_u["KOSPI"].rolling(20, min_periods=10).corr(ret_u["NASDAQ"])
    variants["roll20_uslag_me"] = np.array(
        [month_end_of_series(roll20u, ym) for ym in YMS]
    )

    # weekly US-lag monthly
    wlag = weekly[["KOSPI", "NASDAQ"]].copy()
    wlag["NASDAQ"] = wlag["NASDAQ"].shift(1)
    wlr = wlag.pct_change(fill_method=None)
    wlr["YM"] = wlr.index.to_period("M").astype(str)
    variants["weekly_uslag1_monthly"] = np.array(
        [
            float(g["KOSPI"].corr(g["NASDAQ"]))
            if len(g := wlr.loc[wlr["YM"] == ym, ["KOSPI", "NASDAQ"]].dropna()) >= 2
            else np.nan
            for ym in YMS
        ]
    )

    # log returns weekly monthly
    wlog = np.log(weekly[["KOSPI", "NASDAQ"]]).diff()
    wlog["YM"] = wlog.index.to_period("M").astype(str)
    variants["weekly_logret_monthly"] = np.array(
        [
            float(g["KOSPI"].corr(g["NASDAQ"]))
            if len(g := wlog.loc[wlog["YM"] == ym, ["KOSPI", "NASDAQ"]].dropna()) >= 2
            else np.nan
            for ym in YMS
        ]
    )

    # price-level weekly monthly corr (not returns) — sometimes people do this by mistake
    wp = weekly[["KOSPI", "NASDAQ"]].copy()
    wp["YM"] = wp.index.to_period("M").astype(str)
    variants["weekly_pricelevel_monthly"] = np.array(
        [
            float(g["KOSPI"].corr(g["NASDAQ"]))
            if len(g := wp.loc[wp["YM"] == ym, ["KOSPI", "NASDAQ"]].dropna()) >= 2
            else np.nan
            for ym in YMS
        ]
    )

    # 20d rolling on log returns ME
    logret = np.log(daily[["KOSPI", "NASDAQ"]]).diff()
    roll20log = logret["KOSPI"].rolling(20, min_periods=10).corr(logret["NASDAQ"])
    variants["roll20_log_me"] = np.array(
        [month_end_of_series(roll20log, ym) for ym in YMS]
    )

    # month-mean of 20d rolling (not month-end)
    variants["roll20_month_mean"] = np.array(
        [
            float(m.mean())
            if len(
                m := roll20.loc[roll20.index.to_period("M").astype(str) == ym].dropna()
            )
            else np.nan
            for ym in YMS
        ]
    )

    print("=" * 72)
    print("RANKING by MAE vs original table")
    print("=" * 72)
    rank = []
    for name, vals in variants.items():
        mae, rmse = mae_rmse(vals)
        rank.append((mae, rmse, name, vals))
    rank.sort(key=lambda x: (x[0], x[1]))
    print(f"{'method':32s} {'MAE':>8s} {'RMSE':>8s}")
    for mae, rmse, name, _ in rank:
        print(f"{name:32s} {mae:8.4f} {rmse:8.4f}")

    best_mae, best_rmse, best_name, best_vals = rank[0]
    print()
    print(f"BEST = {best_name}  (MAE={best_mae:.4f}, RMSE={best_rmse:.4f})")
    print()

    show = pd.DataFrame({"YM": YMS, "orig": TARGETS})
    for _, _, name, vals in rank[:5]:
        show[name] = np.round(vals, 4)
        show["d_" + name[:14]] = np.round(vals - TARGETS, 4)
    print("=" * 72)
    print("TOP-5 methods detail vs orig (d_* = method - orig)")
    print("=" * 72)
    print(show.to_string(index=False))

    out_path = ROOT / "data/processed/corr_method_comparison.csv"
    show.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n[SAVE] {out_path}")

    choice_path = ROOT / "data/processed/corr_method_choice.txt"
    choice_path.write_text(
        f"best_method={best_name}\nMAE={best_mae:.6f}\nRMSE={best_rmse:.6f}\n",
        encoding="utf-8",
    )
    print(f"[SAVE] {choice_path}")
    print(best_name)
    return best_name


if __name__ == "__main__":
    main()

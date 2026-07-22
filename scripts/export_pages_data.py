"""
processed CSV → docs/data/*.json (GitHub Pages용 '서버리스 DB')
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "data"


def _json_safe(obj):
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if pd.isna(obj):
        return None
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


def series_frame(df: pd.DataFrame, cols: list[str]) -> dict:
    out = {"date": [d.strftime("%Y-%m-%d") for d in df.index]}
    for c in cols:
        if c not in df.columns:
            out[c] = [None] * len(df)
            continue
        out[c] = [_json_safe(v) for v in pd.to_numeric(df[c], errors="coerce")]
    return out


def table_records(df: pd.DataFrame) -> list[dict]:
    recs = []
    for _, row in df.iterrows():
        recs.append({k: _json_safe(v) for k, v in row.items()})
    return recs


def dump(name: str, payload: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {path}")


def main() -> None:
    meta_path = PROCESSED / "corr_method_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    tab1 = pd.read_csv(PROCESSED / "corr_rolling_tab1.csv", index_col=0, parse_dates=True)
    panel2 = pd.read_csv(PROCESSED / "panel2_rrp_sofr.csv", index_col=0, parse_dates=True)
    panel3 = pd.read_csv(PROCESSED / "panel3_korea_alpha.csv", index_col=0, parse_dates=True)

    dump(
        "panel1.json",
        series_frame(
            tab1,
            [
                "corr_weekly_roll8_ffill",
                "corr_daily_roll30",
                "fed_net_liquidity_weekly",
                "KOSPI_rebased",
                "NASDAQ_rebased",
            ],
        ),
    )
    dump(
        "panel2.json",
        series_frame(
            panel2,
            [
                "RRPONTSYD",
                "sofr_iorb_spread",
                "sofr_iorb_vol20",
                "fed_net_liquidity",
                "fed_net_liquidity_weekly",
                "rrp_depleted",
                "is_month_end",
                "is_quarter_end",
            ],
        ),
    )
    dump(
        "panel3.json",
        series_frame(
            panel3,
            [
                "KOSPI_rebased",
                "EEM_rebased",
                "foreign_netbuy",
                "korea_alpha",
            ],
        ),
    )

    for src, dst in [
        ("corr_monthly_adopted.csv", "monthly_corr.json"),
        ("rrp_regime_corr.csv", "rrp_regime.json"),
        ("rrp_spread_corr_summary.csv", "rrp_summary.json"),
        ("korea_alpha_monthly_corr.csv", "korea_alpha_monthly.json"),
        ("korea_alpha_corr_summary.csv", "korea_alpha_summary.json"),
        ("corr_method_A_vs_B.csv", "method_ab.json"),
    ]:
        p = PROCESSED / src
        if p.exists():
            df = pd.read_csv(p)
            dump(dst, table_records(df))

    # Lead-lag from master
    master_p = PROCESSED / "master_dataset.csv"
    lead_lag = []
    if master_p.exists():
        m = pd.read_csv(master_p, index_col=0, parse_dates=True)
        k = pd.to_numeric(m.get("KOSPI"), errors="coerce").pct_change(fill_method=None)
        n = pd.to_numeric(m.get("NASDAQ"), errors="coerce").pct_change(fill_method=None)
        if "is_kr_trading_day" in m.columns:
            mask = m["is_kr_trading_day"].fillna(False).astype(bool)
            k = k.where(mask)
            n = n.where(mask)
        for lag in range(-5, 6):
            if lag >= 0:
                x, y = k, n.shift(lag)
                label = f"NASDAQ 선행 {lag}일" if lag else "동시 (lag=0)"
            else:
                x, y = k.shift(-lag), n
                label = f"KOSPI 선행 {-lag}일"
            both = pd.concat([x, y], axis=1).dropna()
            corr = float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) >= 20 else None
            lead_lag.append({"lag": lag, "label": label, "corr": corr, "n": int(len(both))})
    dump("lead_lag.json", lead_lag)
    dump("meta.json", meta)
    dump(
        "manifest.json",
        {
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "files": sorted(p.name for p in OUT.glob("*.json")),
        },
    )
    print("[OK] pages data export complete")


if __name__ == "__main__":
    main()

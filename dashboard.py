"""
최종 Streamlit 대시보드 — 글로벌 달러 유동성 헤게모니 · KOSPI–NASDAQ

탭 구성
  0. 요약 — 핵심 판정 / KPI
  1. Panel 1 — 주간 8주 롤링 상관 + Fed 순유동성 배경
  2. Panel 2 — Fed 순유동성 · SOFR−IORB · RRP 고갈 음영 · 월말/분기말 마커
  3. Panel 3 — KOSPI/EEM 리베이스 + 일별 foreign_netbuy
  4. 방법론·RRP 검증 — Method A/B, 캘린더 제외 상관
  5. Lead-Lag — KOSPI vs NASDAQ 시차 상관
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"

EVENTS = {
    "2026-06-18": "FOMC",
}

PLOTLY_CFG = {"displayModeBar": True, "responsive": True}


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


@st.cache_data
def load_table(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_meta() -> dict:
    path = PROCESSED / "corr_method_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def add_event_vlines(fig: go.Figure) -> None:
    for date_str, label in EVENTS.items():
        if "X" in date_str:
            continue
        try:
            ts = pd.Timestamp(date_str)
        except Exception:
            continue
        fig.add_vline(
            x=ts,
            line_width=1,
            line_dash="dot",
            line_color="rgba(120,120,120,0.8)",
            annotation_text=label,
            annotation_position="top left",
        )


def coerce_bool_series(s: pd.Series) -> pd.Series:
    """CSV에서 True/False가 문자열로 읽혀도 안전하게 bool 변환."""
    if s.dtype == bool:
        return s
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(bool)
    return s.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "t", "y"})


def depleted_spans(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if mask.empty:
        return []
    m = coerce_bool_series(mask).fillna(False)
    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    prev = None
    for ts, flag in m.items():
        if flag and start is None:
            start = pd.Timestamp(ts)
        if flag:
            prev = pd.Timestamp(ts)
        elif start is not None:
            spans.append((start, prev if prev is not None else start))
            start = None
            prev = None
    if start is not None:
        spans.append((start, prev if prev is not None else start))
    return spans


def add_depleted_shapes(
    fig: go.Figure,
    mask: pd.Series,
    y_ref: pd.Series | None = None,
) -> None:
    """
    RRP 고갈 구간 회색 음영.
    Streamlit+Plotly에서 layout shapes가 종종 안 보여, fill scatter로 그림.
    """
    spans = depleted_spans(mask)
    if not spans:
        return

    if y_ref is not None and y_ref.notna().any():
        y_lo = float(y_ref.min())
        y_hi = float(y_ref.max())
        pad = (y_hi - y_lo) * 0.03 if y_hi > y_lo else 1.0
        y_lo, y_hi = y_lo - pad, y_hi + pad
    else:
        y_lo, y_hi = 0.0, 1.0

    first = True
    for x0, x1 in spans:
        x0 = pd.Timestamp(x0)
        x1 = pd.Timestamp(x1)
        if x1 <= x0:
            x1 = x0 + pd.Timedelta(days=1)
        else:
            x1 = x1 + pd.Timedelta(hours=18)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y_lo, y_lo, y_hi, y_hi, y_lo],
                fill="toself",
                fillcolor="rgba(110, 110, 110, 0.28)",
                line=dict(width=0, color="rgba(0,0,0,0)"),
                mode="lines",
                name="RRP < 10B (음영)",
                showlegend=first,
                hoverinfo="skip",
                legendgroup="rrp_depleted",
            ),
            secondary_y=False,
        )
        first = False


def add_calendar_markers(fig: go.Figure, panel: pd.DataFrame) -> None:
    """월말·분기말 마커 (Panel 2)."""
    if "is_quarter_end" in panel.columns:
        q = panel.index[panel["is_quarter_end"].fillna(False).astype(bool)]
        if len(q):
            yref = panel.get("sofr_iorb_spread")
            y = yref.reindex(q) if yref is not None else pd.Series(0, index=q)
            fig.add_trace(
                go.Scatter(
                    x=q,
                    y=y,
                    mode="markers",
                    name="분기말 (3BD)",
                    marker=dict(symbol="diamond", size=9, color="#7c2d12"),
                    hovertemplate="분기말 %{x|%Y-%m-%d}<extra></extra>",
                ),
                secondary_y=True,
            )
    if "is_month_end" in panel.columns:
        me = panel["is_month_end"].fillna(False).astype(bool)
        if "is_quarter_end" in panel.columns:
            me = me & ~panel["is_quarter_end"].fillna(False).astype(bool)
        m = panel.index[me]
        if len(m):
            yref = panel["sofr_iorb_spread"] if "sofr_iorb_spread" in panel.columns else None
            y = yref.reindex(m) if yref is not None else pd.Series(0.0, index=m)
            fig.add_trace(
                go.Scatter(
                    x=m,
                    y=y,
                    mode="markers",
                    name="월말 (3BD)",
                    marker=dict(symbol="circle-open", size=7, color="#78716c"),
                    hovertemplate="월말 %{x|%Y-%m-%d}<extra></extra>",
                ),
                secondary_y=True,
            )


def slice_range(df: pd.DataFrame, date_range) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        return df.loc[start:end]
    return df


def compute_lead_lag(master: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """lag>0: NASDAQ가 lag일 선행 (어제 NASDAQ vs 오늘 KOSPI)."""
    need = ["KOSPI", "NASDAQ"]
    if any(c not in master.columns for c in need):
        return pd.DataFrame()
    k = pd.to_numeric(master["KOSPI"], errors="coerce").pct_change(fill_method=None)
    n = pd.to_numeric(master["NASDAQ"], errors="coerce").pct_change(fill_method=None)
    if "is_kr_trading_day" in master.columns:
        mask = master["is_kr_trading_day"].fillna(False).astype(bool)
        k = k.where(mask)
        n = n.where(mask)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = k, n.shift(lag)
            label = f"NASDAQ 선행 {lag}일" if lag else "동시 (lag=0)"
        else:
            x, y = k.shift(-lag), n
            label = f"KOSPI 선행 {-lag}일"
        both = pd.concat([x, y], axis=1).dropna()
        corr = float(both.iloc[:, 0].corr(both.iloc[:, 1])) if len(both) >= 20 else np.nan
        rows.append({"lag": lag, "label": label, "corr": corr, "n": len(both)})
    return pd.DataFrame(rows)


def render_overview(meta: dict, regime: pd.DataFrame, alpha_s: pd.DataFrame, monthly: pd.DataFrame) -> None:
    st.subheader("핵심 판정")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("상관 방법론", f"Method {meta.get('adopted_code', 'A')}")
        st.caption(meta.get("adopted_method", ""))
        if meta.get("mae_A") is not None:
            st.caption(f"MAE A={meta['mae_A']:.3f} · B={meta.get('mae_B', float('nan')):.3f}")
    with c2:
        r2 = regime[regime["regime"].astype(str).str.contains("구간2 (2025-11", regex=False)]
        r2x = regime[regime["regime"].astype(str).str.contains("월말·분기말 제외", regex=False)]
        c_before = float(r2["corr_rrp_vs_spread"].iloc[0]) if not r2.empty else np.nan
        c_after = float(r2x["corr_rrp_vs_spread"].iloc[0]) if not r2x.empty else np.nan
        st.metric("구간2 RRP↔spread", f"{c_before:+.3f}" if pd.notna(c_before) else "—")
        st.caption(f"월말·분기말 제외 후 {c_after:+.3f}" if pd.notna(c_after) else "")
    with c3:
        if not alpha_s.empty and "corr" in alpha_s.columns:
            top = alpha_s.iloc[0]
            st.metric("최강 설명 조합 |corr|", f"{float(top['abs_corr']):.3f}")
            st.caption(str(top.get("relationship", "")))
        else:
            st.metric("최강 설명 조합", "—")

    if meta.get("rrp_regime_verdict"):
        st.success(f"**RRP 레짐:** {meta['rrp_regime_verdict']}")
    if meta.get("korea_alpha_verdict"):
        st.info(f"**코리아 알파:** {meta['korea_alpha_verdict']}")

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("**월별 KOSPI–NASDAQ (Method A)**")
        if not monthly.empty:
            cols = [c for c in ["corr", "n_weeks", "corr_label"] if c in monthly.columns]
            st.dataframe(monthly[cols] if cols else monthly, use_container_width=True, height=320)
        else:
            st.caption("corr_monthly_adopted.csv 없음")
    with right:
        st.markdown("**구간별 RRP ↔ SOFR−IORB**")
        if not regime.empty:
            show = regime[
                [
                    c
                    for c in [
                        "regime",
                        "corr_rrp_vs_spread",
                        "n_level",
                        "n_excluded",
                        "rrp_mean",
                    ]
                    if c in regime.columns
                ]
            ]
            st.dataframe(show, use_container_width=True, height=320)
        else:
            st.caption("rrp_regime_corr.csv 없음")


def render_panel1(v1: pd.DataFrame, show_daily30: bool) -> None:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if "fed_net_liquidity_weekly" in v1.columns:
        fed = v1["fed_net_liquidity_weekly"]
        fmin, fmax = fed.min(), fed.max()
        fed_norm = (fed - fmin) / (fmax - fmin) if fmax > fmin else fed * 0
        fig.add_trace(
            go.Scatter(
                x=v1.index,
                y=fed_norm,
                name="Fed 순유동성 (주간, 정규화 배경)",
                fill="tozeroy",
                line=dict(width=0),
                fillcolor="rgba(100,140,180,0.18)",
                hovertemplate="Fed순유동성(raw)=%{customdata:,.0f}<extra></extra>",
                customdata=fed,
            ),
            secondary_y=True,
        )
    fig.add_trace(
        go.Scatter(
            x=v1.index,
            y=v1["corr_weekly_roll8_ffill"],
            name="주간수익률 8주 롤링 상관",
            line=dict(width=2.5, color="#1f4e79"),
        ),
        secondary_y=False,
    )
    if show_daily30 and "corr_daily_roll30" in v1.columns:
        fig.add_trace(
            go.Scatter(
                x=v1.index,
                y=v1["corr_daily_roll30"],
                name="일별 30일 롤링 (참고)",
                line=dict(width=1.2, color="#888888", dash="dot"),
            ),
            secondary_y=False,
        )
    for col, color, name in [
        ("KOSPI_rebased", "rgba(200,80,80,0.30)", "KOSPI (rebased)"),
        ("NASDAQ_rebased", "rgba(80,120,200,0.30)", "NASDAQ (rebased)"),
    ]:
        if col in v1.columns:
            fig.add_trace(
                go.Scatter(
                    x=v1.index,
                    y=v1[col],
                    name=name,
                    line=dict(width=1, color=color),
                ),
                secondary_y=True,
            )
    add_event_vlines(fig)
    fig.update_yaxes(title_text="상관계수", secondary_y=False, range=[-1.05, 1.05])
    fig.update_yaxes(title_text="배경·리베이스", secondary_y=True, showgrid=False)
    fig.update_layout(
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        title="Panel 1 · KOSPI–NASDAQ 롤링 상관",
        margin=dict(t=60),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)

    monthly = load_table("corr_monthly_adopted.csv")
    if not monthly.empty:
        st.caption("* = n_weeks≤4 (표본 부족)")
        cols = [c for c in ["corr", "n_weeks", "low_n", "corr_label"] if c in monthly.columns]
        st.dataframe(monthly[cols] if cols else monthly, use_container_width=True)


def render_panel2(v2: pd.DataFrame, meta: dict) -> None:
    if v2.empty:
        st.warning("panel2_rrp_sofr.csv 없음 — `python correlation_analysis.py` 재실행")
        return

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])

    fed_col = (
        "fed_net_liquidity_weekly"
        if "fed_net_liquidity_weekly" in v2.columns
        else "fed_net_liquidity"
    )
    fed_y = pd.to_numeric(v2[fed_col], errors="coerce")

    # 음영을 먼저 깔아 선 아래에 보이도록
    if "rrp_depleted" in v2.columns:
        dep_mask = coerce_bool_series(v2["rrp_depleted"])
    elif "RRPONTSYD" in v2.columns:
        dep_mask = pd.to_numeric(v2["RRPONTSYD"], errors="coerce") < 10.0
    else:
        dep_mask = pd.Series(False, index=v2.index)
    add_depleted_shapes(fig2, dep_mask, y_ref=fed_y)

    fig2.add_trace(
        go.Scatter(
            x=v2.index,
            y=fed_y,
            name="Fed 순유동성 (주간 ffill)",
            line=dict(width=2.2, color="#1f4e79"),
        ),
        secondary_y=False,
    )
    if "sofr_iorb_vol20" in v2.columns:
        fig2.add_trace(
            go.Scatter(
                x=v2.index,
                y=v2["sofr_iorb_vol20"],
                name="SOFR−IORB 20일 σ",
                fill="tozeroy",
                line=dict(width=0),
                fillcolor="rgba(180,120,60,0.18)",
            ),
            secondary_y=True,
        )
    fig2.add_trace(
        go.Scatter(
            x=v2.index,
            y=v2["sofr_iorb_spread"],
            name="SOFR−IORB 스프레드",
            line=dict(width=2.0, color="#b45309"),
        ),
        secondary_y=True,
    )
    add_calendar_markers(fig2, v2)
    add_event_vlines(fig2)

    fig2.update_yaxes(title_text="Fed 순유동성 (백만 USD)", secondary_y=False)
    fig2.update_yaxes(title_text="SOFR−IORB / σ", secondary_y=True)
    fig2.update_layout(
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        title="Panel 2 · 회색 음영 = RRP < 10B · ◆분기말 · ○월말",
        margin=dict(t=60),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CFG)

    if "RRPONTSYD" in v2.columns:
        with st.expander("RRP 잔고 (참고)"):
            fig_rrp = go.Figure(
                go.Scatter(
                    x=v2.index,
                    y=v2["RRPONTSYD"],
                    name="RRPONTSYD",
                    line=dict(color="#44403c", width=1.8),
                )
            )
            fig_rrp.add_hline(y=10, line_dash="dash", line_color="#a8a29e", annotation_text="10B")
            fig_rrp.update_layout(height=260, margin=dict(t=20, b=20), hovermode="x unified")
            st.plotly_chart(fig_rrp, use_container_width=True, config=PLOTLY_CFG)

    regime = load_table("rrp_regime_corr.csv")
    if not regime.empty:
        st.subheader("구간별 RRP ↔ SOFR−IORB")
        st.dataframe(regime, use_container_width=True)
        if meta.get("rrp_regime_verdict"):
            st.success(f"판정: {meta['rrp_regime_verdict']}")

    rrp_sum = load_table("rrp_spread_corr_summary.csv")
    if not rrp_sum.empty:
        st.subheader("전체 기간 관계 요약")
        st.dataframe(rrp_sum, use_container_width=True)


def render_panel3(v3: pd.DataFrame, meta: dict) -> None:
    if v3.empty:
        st.warning("panel3_korea_alpha.csv 없음 — EEM 수집 후 분석 재실행")
        return

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(
        go.Scatter(
            x=v3.index,
            y=v3["KOSPI_rebased"],
            name="KOSPI (rebased)",
            line=dict(width=2.2, color="#c0392b"),
        ),
        secondary_y=False,
    )
    fig3.add_trace(
        go.Scatter(
            x=v3.index,
            y=v3["EEM_rebased"],
            name="EEM (rebased)",
            line=dict(width=2.0, color="#2980b9"),
        ),
        secondary_y=False,
    )
    fig3.add_trace(
        go.Bar(
            x=v3.index,
            y=v3["foreign_netbuy"],
            name="외국인 순매수 (일별, 백만원)",
            marker_color="rgba(120,120,120,0.45)",
        ),
        secondary_y=True,
    )
    add_event_vlines(fig3)
    fig3.update_yaxes(title_text="지수 (2025-08-01=100)", secondary_y=False)
    fig3.update_yaxes(title_text="foreign_netbuy", secondary_y=True)
    fig3.update_layout(
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
        title="Panel 3 · KOSPI vs EEM 커플링 + 일별 외국인 순매수",
        barmode="relative",
        margin=dict(t=60),
    )
    st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CFG)

    if "korea_alpha" in v3.columns:
        with st.expander("코리아 알파 (KOSPI_ret − EEM_ret) 시계열"):
            fig_a = go.Figure(
                go.Bar(
                    x=v3.index,
                    y=v3["korea_alpha"],
                    name="korea_alpha",
                    marker_color="rgba(80,80,80,0.5)",
                )
            )
            fig_a.update_layout(height=260, margin=dict(t=20, b=20), hovermode="x unified")
            st.plotly_chart(fig_a, use_container_width=True, config=PLOTLY_CFG)

    alpha_m = load_table("korea_alpha_monthly_corr.csv")
    if not alpha_m.empty:
        st.subheader("월별 상관: KOSPI↔EEM | KOSPI↔DXY | alpha↔Fed")
        st.dataframe(alpha_m, use_container_width=True)
    alpha_s = load_table("korea_alpha_corr_summary.csv")
    if not alpha_s.empty:
        st.subheader("전체 기간 관계 요약 (|corr| 내림차순)")
        st.dataframe(alpha_s, use_container_width=True)
    if meta.get("korea_alpha_verdict"):
        st.success(meta["korea_alpha_verdict"])


def render_method(meta: dict) -> None:
    cmp_df = load_table("corr_method_A_vs_B.csv")
    if not cmp_df.empty:
        st.subheader("원본 표 vs Method A / B")
        st.dataframe(cmp_df, use_container_width=True)
        if meta:
            st.caption(
                f"채택: Method {meta.get('adopted_code')} ({meta.get('adopted_method')}) · "
                f"MAE_A={meta.get('mae_A', float('nan')):.4f}, "
                f"MAE_B={meta.get('mae_B', float('nan')):.4f}"
            )

    regime = load_table("rrp_regime_corr.csv")
    if not regime.empty:
        st.subheader("RRP 여유 vs 고갈 + 캘린더 제외")
        st.dataframe(regime, use_container_width=True)
        # 전후 비교 카드
        r2 = regime[regime["regime"].astype(str).str.contains("구간2 (2025-11", regex=False)]
        r2x = regime[regime["regime"].astype(str).str.contains("월말·분기말 제외", regex=False)]
        if not r2.empty and not r2x.empty:
            a, b, c = st.columns(3)
            a.metric("구간2 (포함)", f"{float(r2['corr_rrp_vs_spread'].iloc[0]):+.3f}")
            b.metric(
                "월말·분기말 제외",
                f"{float(r2x['corr_rrp_vs_spread'].iloc[0]):+.3f}",
                delta=f"{float(r2x['corr_rrp_vs_spread'].iloc[0]) - float(r2['corr_rrp_vs_spread'].iloc[0]):+.3f}",
            )
            n_ex = int(r2x["n_excluded"].iloc[0]) if "n_excluded" in r2x.columns else 0
            c.metric("제외 일수", n_ex)
        if meta.get("rrp_regime_verdict"):
            st.success(f"판정: {meta['rrp_regime_verdict']}")

    rrp_sum = load_table("rrp_spread_corr_summary.csv")
    if not rrp_sum.empty:
        st.subheader("RRP vs Fed순유동성 — 스프레드 관계 강도")
        st.dataframe(rrp_sum, use_container_width=True)


def render_lead_lag(master: pd.DataFrame, date_range) -> None:
    if master.empty:
        st.warning("master_dataset.csv 없음")
        return
    view = slice_range(master, date_range)
    ll = compute_lead_lag(view, max_lag=5)
    if ll.empty:
        st.warning("Lead-Lag 계산 불가 (KOSPI/NASDAQ 필요)")
        return

    best = ll.loc[ll["corr"].abs().idxmax()]
    st.caption(
        f"선택 기간 최강: **{best['label']}** (corr={best['corr']:+.3f}, n={int(best['n'])}). "
        "lag>0 = NASDAQ 선행."
    )

    fig = go.Figure(
        go.Bar(
            x=ll["lag"],
            y=ll["corr"],
            text=[f"{v:+.3f}" if pd.notna(v) else "" for v in ll["corr"]],
            textposition="outside",
            marker_color=["#1f4e79" if lag == 0 else "#78716c" for lag in ll["lag"]],
            hovertemplate="lag=%{x}<br>corr=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_color="#a8a29e", line_width=1)
    fig.update_layout(
        height=420,
        title="KOSPI_ret vs NASDAQ_ret 시차 상관 (−5 ~ +5일)",
        xaxis_title="lag (일) · + = NASDAQ 선행",
        yaxis_title="Pearson corr",
        yaxis=dict(range=[-0.2, 0.6]),
        margin=dict(t=50),
    )
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
    st.dataframe(ll, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="유동성 헤게모니 · 최종 대시보드",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("글로벌 달러 유동성 헤게모니 · 최종 대시보드")
    st.caption("분석 기간 2025-08 ~ 2026-07 · 산출물: data/processed/")

    meta = load_meta()
    tab1 = load_csv("corr_rolling_tab1.csv")
    panel2 = load_csv("panel2_rrp_sofr.csv")
    panel3 = load_csv("panel3_korea_alpha.csv")
    master = load_csv("master_dataset.csv")
    regime = load_table("rrp_regime_corr.csv")
    alpha_s = load_table("korea_alpha_corr_summary.csv")
    monthly = load_table("corr_monthly_adopted.csv")

    if tab1.empty:
        st.error("`corr_rolling_tab1.csv` 없음. `python correlation_analysis.py` 실행 필요.")
        st.stop()

    with st.sidebar:
        st.header("필터")
        date_range = st.date_input(
            "기간",
            value=(tab1.index.min().date(), tab1.index.max().date()),
        )
        show_daily30 = st.checkbox("Panel1: 일별 30일 롤링(참고)", value=False)
        st.markdown("---")
        st.markdown("**패널 가이드**")
        st.markdown(
            "- **Panel 1** 주간 8주 롤링 상관\n"
            "- **Panel 2** Fed 유동성 · SOFR−IORB · RRP 고갈\n"
            "- **Panel 3** KOSPI/EEM + 외국인 순매수\n"
            "- **검증** Method A + 캘린더 제외\n"
            "- **Lead-Lag** NASDAQ 선행성"
        )
        if meta.get("adopted_code"):
            st.markdown("---")
            st.markdown(f"채택 방법: **Method {meta['adopted_code']}**")

    v1 = slice_range(tab1, date_range)
    v2 = slice_range(panel2, date_range)
    v3 = slice_range(panel3, date_range)

    tabs = st.tabs(
        [
            "요약",
            "Panel 1 · 롤링 상관",
            "Panel 2 · 유동성",
            "Panel 3 · 코리아 알파",
            "방법론·RRP 검증",
            "Lead-Lag",
        ]
    )

    with tabs[0]:
        render_overview(meta, regime, alpha_s, monthly)
    with tabs[1]:
        render_panel1(v1, show_daily30)
    with tabs[2]:
        render_panel2(v2, meta)
    with tabs[3]:
        render_panel3(v3, meta)
    with tabs[4]:
        render_method(meta)
    with tabs[5]:
        render_lead_lag(master, date_range)


if __name__ == "__main__":
    main()

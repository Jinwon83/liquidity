/* global Plotly */
const DATA = "data";

async function loadJSON(name) {
  const res = await fetch(`${DATA}/${name}?t=${Date.now()}`);
  if (!res.ok) throw new Error(`${name} ${res.status}`);
  return res.json();
}

function tableHTML(rows, cols) {
  if (!rows || !rows.length) return "<p>데이터 없음</p>";
  const keys = cols || Object.keys(rows[0]);
  const head = keys.map((k) => `<th>${k}</th>`).join("");
  const body = rows
    .map((r) => {
      const tds = keys
        .map((k) => {
          let v = r[k];
          if (typeof v === "number" && Number.isFinite(v)) v = v.toFixed(4);
          if (v === null || v === undefined) v = "";
          return `<td>${v}</td>`;
        })
        .join("");
      return `<tr>${tds}</tr>`;
    })
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function layoutBase(title) {
  return {
    title: { text: title, font: { size: 14 } },
    margin: { t: 48, r: 50, b: 40, l: 55 },
    legend: { orientation: "h", y: 1.12 },
    hovermode: "x unified",
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#ffffff",
    font: { color: "#1c1917", size: 11 },
  };
}

function depletedShapes(dates, flags) {
  const shapes = [];
  let start = null;
  for (let i = 0; i < dates.length; i++) {
    const on = !!flags[i];
    if (on && start === null) start = dates[i];
    if (!on && start !== null) {
      shapes.push({
        type: "rect",
        xref: "x",
        yref: "paper",
        x0: start,
        x1: dates[i - 1] || start,
        y0: 0,
        y1: 1,
        fillcolor: "rgba(110,110,110,0.22)",
        line: { width: 0 },
        layer: "below",
      });
      start = null;
    }
  }
  if (start !== null) {
    shapes.push({
      type: "rect",
      xref: "x",
      yref: "paper",
      x0: start,
      x1: dates[dates.length - 1],
      y0: 0,
      y1: 1,
      fillcolor: "rgba(110,110,110,0.22)",
      line: { width: 0 },
      layer: "below",
    });
  }
  return shapes;
}

function renderKPIs(meta, regime, alphaSum) {
  const el = document.getElementById("kpis");
  const r2 = (regime || []).find((r) => String(r.regime || "").includes("구간2 (2025-11"));
  const r2x = (regime || []).find((r) => String(r.regime || "").includes("월말·분기말 제외"));
  const top = (alphaSum || [])[0];
  el.innerHTML = `
    <div class="kpi">
      <div class="label">상관 방법론</div>
      <div class="value">Method ${meta.adopted_code || "A"}</div>
      <div class="sub">${meta.adopted_method || ""}</div>
    </div>
    <div class="kpi">
      <div class="label">구간2 RRP↔spread</div>
      <div class="value">${r2 && r2.corr_rrp_vs_spread != null ? (r2.corr_rrp_vs_spread > 0 ? "+" : "") + Number(r2.corr_rrp_vs_spread).toFixed(3) : "—"}</div>
      <div class="sub">월말·분기말 제외 ${r2x && r2x.corr_rrp_vs_spread != null ? (Number(r2x.corr_rrp_vs_spread) > 0 ? "+" : "") + Number(r2x.corr_rrp_vs_spread).toFixed(3) : "—"}</div>
    </div>
    <div class="kpi">
      <div class="label">최강 |corr|</div>
      <div class="value">${top && top.abs_corr != null ? Number(top.abs_corr).toFixed(3) : "—"}</div>
      <div class="sub">${top ? top.relationship : ""}</div>
    </div>`;
}

function renderPanel1(p1) {
  const dates = p1.date;
  const traces = [
    {
      x: dates,
      y: p1.corr_weekly_roll8_ffill,
      name: "주간 8주 롤링 상관",
      line: { color: "#1f4e79", width: 2.4 },
    },
  ];
  if (p1.KOSPI_rebased) {
    traces.push({
      x: dates,
      y: p1.KOSPI_rebased,
      name: "KOSPI rebased",
      yaxis: "y2",
      line: { color: "rgba(200,80,80,0.35)", width: 1 },
    });
  }
  if (p1.NASDAQ_rebased) {
    traces.push({
      x: dates,
      y: p1.NASDAQ_rebased,
      name: "NASDAQ rebased",
      yaxis: "y2",
      line: { color: "rgba(80,120,200,0.35)", width: 1 },
    });
  }
  const layout = {
    ...layoutBase("Panel 1 · KOSPI–NASDAQ 롤링 상관"),
    yaxis: { title: "상관", range: [-1.05, 1.05] },
    yaxis2: { title: "리베이스", overlaying: "y", side: "right", showgrid: false },
  };
  Plotly.newPlot("chart-p1", traces, layout, { responsive: true, displayModeBar: false });
}

function renderPanel2(p2) {
  const dates = p2.date;
  const shapes = depletedShapes(dates, p2.rrp_depleted || []);
  const traces = [
    {
      x: dates,
      y: p2.fed_net_liquidity_weekly || p2.fed_net_liquidity,
      name: "Fed 순유동성",
      line: { color: "#1f4e79", width: 2.2 },
    },
    {
      x: dates,
      y: p2.sofr_iorb_spread,
      name: "SOFR−IORB",
      yaxis: "y2",
      line: { color: "#b45309", width: 2 },
    },
  ];
  if (p2.sofr_iorb_vol20) {
    traces.splice(1, 0, {
      x: dates,
      y: p2.sofr_iorb_vol20,
      name: "σ20",
      yaxis: "y2",
      fill: "tozeroy",
      line: { width: 0 },
      fillcolor: "rgba(180,120,60,0.18)",
    });
  }
  const layout = {
    ...layoutBase("Panel 2 · 회색=RRP<10B"),
    shapes,
    yaxis: { title: "Fed 순유동성" },
    yaxis2: { title: "SOFR−IORB", overlaying: "y", side: "right" },
  };
  Plotly.newPlot("chart-p2", traces, layout, { responsive: true, displayModeBar: false });
}

function renderPanel3(p3) {
  const dates = p3.date;
  const traces = [
    {
      x: dates,
      y: p3.KOSPI_rebased,
      name: "KOSPI",
      line: { color: "#c0392b", width: 2.2 },
    },
    {
      x: dates,
      y: p3.EEM_rebased,
      name: "EEM",
      line: { color: "#2980b9", width: 2 },
    },
    {
      x: dates,
      y: p3.foreign_netbuy,
      name: "외국인 순매수",
      type: "bar",
      yaxis: "y2",
      marker: { color: "rgba(120,120,120,0.45)" },
    },
  ];
  const layout = {
    ...layoutBase("Panel 3 · KOSPI/EEM + 외국인 순매수"),
    barmode: "relative",
    yaxis: { title: "리베이스 지수" },
    yaxis2: { title: "순매수", overlaying: "y", side: "right" },
  };
  Plotly.newPlot("chart-p3", traces, layout, { responsive: true, displayModeBar: false });
}

function renderLeadLag(rows) {
  Plotly.newPlot(
    "chart-ll",
    [
      {
        x: rows.map((r) => r.lag),
        y: rows.map((r) => r.corr),
        type: "bar",
        marker: {
          color: rows.map((r) => (r.lag === 0 ? "#1f4e79" : "#78716c")),
        },
        text: rows.map((r) => (r.corr == null ? "" : r.corr.toFixed(3))),
        textposition: "outside",
      },
    ],
    {
      ...layoutBase("KOSPI vs NASDAQ 시차 상관"),
      xaxis: { title: "lag (+ = NASDAQ 선행)" },
      yaxis: { title: "corr", range: [-0.2, 0.6] },
    },
    { responsive: true, displayModeBar: false }
  );
  document.getElementById("tbl-ll").innerHTML = tableHTML(rows);
}

function bindTabs() {
  const tabs = document.querySelectorAll(".tab");
  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabs.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
      document.getElementById(btn.dataset.tab).classList.add("active");
      window.dispatchEvent(new Event("resize"));
    });
  });
}

async function main() {
  bindTabs();
  const [
    meta,
    manifest,
    p1,
    p2,
    p3,
    monthly,
    regime,
    alphaM,
    alphaSum,
    method,
    leadLag,
  ] = await Promise.all([
    loadJSON("meta.json"),
    loadJSON("manifest.json"),
    loadJSON("panel1.json"),
    loadJSON("panel2.json"),
    loadJSON("panel3.json"),
    loadJSON("monthly_corr.json"),
    loadJSON("rrp_regime.json"),
    loadJSON("korea_alpha_monthly.json"),
    loadJSON("korea_alpha_summary.json"),
    loadJSON("method_ab.json"),
    loadJSON("lead_lag.json"),
  ]);

  renderKPIs(meta, regime, alphaSum);
  document.getElementById("verdict-rrp").textContent =
    "RRP 레짐: " + (meta.rrp_regime_verdict || "—");
  document.getElementById("verdict-alpha").textContent =
    "코리아 알파: " + (meta.korea_alpha_verdict || "—");
  document.getElementById("tbl-monthly").innerHTML =
    "<h3>월별 KOSPI–NASDAQ (Method A)</h3>" + tableHTML(monthly);
  document.getElementById("tbl-regime").innerHTML =
    "<h3>구간별 RRP ↔ SOFR−IORB</h3>" + tableHTML(regime);
  document.getElementById("tbl-regime2").innerHTML = tableHTML(regime);
  document.getElementById("tbl-method").innerHTML =
    "<h3>Method A vs B</h3>" + tableHTML(method);
  document.getElementById("tbl-alpha-m").innerHTML =
    "<h3>월별 KOSPI↔EEM / DXY / alpha↔Fed</h3>" + tableHTML(alphaM);

  renderPanel1(p1);
  renderPanel2(p2);
  renderPanel3(p3);
  renderLeadLag(leadLag);

  document.getElementById("footer").textContent =
    `데이터 갱신: ${manifest.generated_at || "—"} · 파일 ${
      (manifest.files || []).length
    }개 · GitHub Pages 서버리스`;
}

main().catch((err) => {
  document.getElementById("footer").textContent = "로드 실패: " + err.message;
  console.error(err);
});

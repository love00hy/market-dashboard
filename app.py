"""
📈 마켓 대시보드
실행:  streamlit run app.py
데이터: Yahoo Finance (yfinance) — 무료, 약 15분 지연
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import analysis as A

st.set_page_config(page_title="마켓 대시보드", page_icon="📈", layout="wide")

# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────
INDICES = {
    "^GSPC": "S&P 500",
    "^IXIC": "나스닥 종합",
    "^NDX": "나스닥 100",
    "^DJI": "다우존스",
    "^RUT": "러셀 2000",
    "^VIX": "VIX(공포지수)",
}

# 섹터 ETF : (한글명, 뉴스 수집용 대표 보유종목)
SECTORS = {
    "XLK": ("기술", ["NVDA", "AAPL", "MSFT"]),
    "XLC": ("커뮤니케이션", ["META", "GOOGL", "NFLX"]),
    "XLY": ("경기소비재", ["AMZN", "TSLA", "HD"]),
    "XLF": ("금융", ["JPM", "V", "BRK-B"]),
    "XLV": ("헬스케어", ["LLY", "UNH", "JNJ"]),
    "XLI": ("산업재", ["GE", "CAT", "RTX"]),
    "XLE": ("에너지", ["XOM", "CVX", "COP"]),
    "XLP": ("필수소비재", ["COST", "WMT", "PG"]),
    "XLU": ("유틸리티", ["NEE", "SO", "DUK"]),
    "XLB": ("소재", ["LIN", "SHW", "FCX"]),
    "XLRE": ("부동산", ["PLD", "AMT", "EQIX"]),
}

PERIODS = {"1일": 1, "1주": 5, "1개월": 21, "3개월": 63, "연초 대비": "YTD"}

UP, DOWN, NEUTRAL = "#26a69a", "#ef5350", "#9e9e9e"


# ─────────────────────────────────────────────
# 데이터 로딩 (캐시)
# ─────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_prices(tickers: tuple[str, ...], period: str = "1y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    raw = yf.download(list(tickers), period=period, interval=interval, group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(how="all")
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                out[t] = df
        except KeyError:
            pass
    return out


def _parse_news_item(item: dict) -> dict | None:
    """yfinance 뉴스 포맷(구/신 버전) 모두 처리."""
    c = item.get("content", item)
    title = c.get("title")
    if not title:
        return None
    url = ((c.get("canonicalUrl") or {}).get("url")
           or (c.get("clickThroughUrl") or {}).get("url")
           or c.get("link"))
    provider = (c.get("provider") or {}).get("displayName") or c.get("publisher", "")
    ts = c.get("pubDate") or c.get("displayTime")
    if ts:
        when = pd.to_datetime(ts, utc=True, errors="coerce")
    elif c.get("providerPublishTime"):
        when = pd.to_datetime(c["providerPublishTime"], unit="s", utc=True)
    else:
        when = pd.NaT
    return {"title": title, "url": url, "provider": provider, "time": when,
            "summary": c.get("summary") or ""}


@st.cache_data(ttl=900, show_spinner=False)
def load_news(tickers: tuple[str, ...], per_ticker: int = 6) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            items = yf.Ticker(t).news or []
        except Exception:
            items = []
        for it in items[:per_ticker]:
            p = _parse_news_item(it)
            if p:
                p["ticker"] = t
                p["sentiment"] = A.headline_sentiment(p["title"] + " " + p["summary"])
                rows.append(p)
    if not rows:
        return pd.DataFrame(columns=["title", "url", "provider", "time", "summary", "ticker", "sentiment"])
    df = pd.DataFrame(rows).drop_duplicates("title")
    return df.sort_values("time", ascending=False, na_position="last")


def period_change(df: pd.DataFrame, key) -> float:
    return A.ytd_change(df["Close"]) if key == "YTD" else A.pct_change_over(df["Close"], key)


def fmt_usd(x: float) -> str:
    if pd.isna(x):
        return "-"
    sign = "+" if x > 0 else "-" if x < 0 else ""
    x = abs(x)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if x >= div:
            return f"{sign}${x / div:,.1f}{unit}"
    return f"{sign}${x:,.0f}"


def ago(ts) -> str:
    if pd.isna(ts):
        return ""
    mins = (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds() / 60
    if mins < 60:
        return f"{mins:.0f}분 전"
    if mins < 60 * 24:
        return f"{mins / 60:.0f}시간 전"
    return f"{mins / 1440:.0f}일 전"


# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    if st.button("🔄 데이터 새로고침", width="stretch"):
        st.cache_data.clear()
    st.caption("데이터는 10분간 캐시됩니다.")
    st.divider()
    st.subheader("🪙 기술적 분석 종목")
    ta_ticker = st.text_input("티커", value="COIN").strip().upper() or "COIN"
    ta_period = st.selectbox("기간", ["3mo", "6mo", "1y", "2y"], index=2,
                             format_func=lambda p: {"3mo": "3개월", "6mo": "6개월", "1y": "1년", "2y": "2년"}[p])
    vp_bins = st.slider("매물대 구간 수", 20, 80, 40, step=5)
    swing_order = st.slider("스윙 감도(작을수록 레벨 많음)", 3, 15, 5)
    st.divider()
    st.caption("⚠️ 투자 참고용입니다. 투자 판단과 책임은 본인에게 있습니다.")

st.title("📈 마켓 대시보드")
st.caption(f"Yahoo Finance 기준 (약 15분 지연) · 조회 시각 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

tab_idx, tab_sec, tab_news, tab_ta = st.tabs(
    ["📊 주요 지수", "🏭 섹터 & 자금흐름", "📰 강세·약세 이유(뉴스)", f"🪙 {ta_ticker} 기술적 분석"])

# ═════════════════════════════════════════════
# 1) 주요 지수
# ═════════════════════════════════════════════
with tab_idx:
    with st.spinner("지수 데이터 불러오는 중…"):
        idx_data = load_prices(tuple(INDICES), "1y")

    cols = st.columns(len(INDICES))
    for col, (t, name) in zip(cols, INDICES.items()):
        df = idx_data.get(t)
        if df is None or len(df) < 2:
            col.metric(name, "-")
            continue
        last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
        chg = (last / prev - 1) * 100
        # VIX는 오르면 나쁜 신호 → 색 반전
        col.metric(name, f"{last:,.2f}", f"{chg:+.2f}%",
                   delta_color="inverse" if t == "^VIX" else "normal")

    c1, c2 = st.columns([3, 2])
    with c1:
        rng = st.radio("비교 기간", ["1개월", "3개월", "6개월", "1년"], index=2, horizontal=True, key="idx_rng")
        n = {"1개월": 21, "3개월": 63, "6개월": 126, "1년": 252}[rng]
        fig = go.Figure()
        for t, name in INDICES.items():
            if t == "^VIX" or t not in idx_data:
                continue
            s = idx_data[t]["Close"].tail(n)
            fig.add_trace(go.Scatter(x=s.index, y=(s / s.iloc[0] - 1) * 100, name=name, mode="lines"))
        fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
        fig.update_layout(title=f"지수 수익률 비교 ({rng}, %)", height=420, hovermode="x unified",
                          yaxis_ticksuffix="%", margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
    with c2:
        sel = st.selectbox("지수 캔들 차트", [t for t in INDICES if t != "^VIX"],
                           format_func=lambda t: INDICES[t])
        if sel in idx_data:
            d = idx_data[sel].tail(126).copy()
            d["MA20"], d["MA50"] = d["Close"].rolling(20).mean(), d["Close"].rolling(50).mean()
            f2 = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                          close=d["Close"], name=INDICES[sel],
                                          increasing_line_color=UP, decreasing_line_color=DOWN))
            for m, color in (("MA20", "#ffb300"), ("MA50", "#42a5f5")):
                f2.add_trace(go.Scatter(x=d.index, y=d[m], name=m, line=dict(width=1.2, color=color)))
            f2.update_layout(height=420, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=50, b=10),
                             title=f"{INDICES[sel]} (6개월)", legend=dict(orientation="h", y=-0.15))
            f2.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(f2, width="stretch")

# ═════════════════════════════════════════════
# 2) 섹터 & 자금흐름
# ═════════════════════════════════════════════
with st.spinner("섹터 데이터 불러오는 중…"):
    sec_data = load_prices(tuple(SECTORS), "1y")


def sector_table(period_key) -> pd.DataFrame:
    rows = []
    for t, (name, _) in SECTORS.items():
        df = sec_data.get(t)
        if df is None:
            continue
        flow = A.money_flow_stats(df, window=5)
        row = {"티커": t, "섹터": name, "선택기간": period_change(df, period_key)}
        for label, key in PERIODS.items():
            row[label] = period_change(df, key)
        row["순자금(5일 추정)"] = flow["net_flow_usd"]
        row["상대거래량"] = flow["rel_volume"]
        row["CMF(20)"] = flow["cmf"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("선택기간", ascending=False).reset_index(drop=True)


with tab_sec:
    p_label = st.radio("기간", list(PERIODS), index=1, horizontal=True, key="sec_p")
    st.session_state["sector_period"] = p_label
    tbl = sector_table(PERIODS[p_label])

    c1, c2 = st.columns(2)
    with c1:
        t_sorted = tbl.sort_values("선택기간")
        fig = go.Figure(go.Bar(
            x=t_sorted["선택기간"], y=t_sorted["섹터"] + " (" + t_sorted["티커"] + ")", orientation="h",
            marker_color=[UP if v >= 0 else DOWN for v in t_sorted["선택기간"]],
            text=[f"{v:+.2f}%" for v in t_sorted["선택기간"]], textposition="outside"))
        fig.update_layout(title=f"섹터별 등락률 ({p_label})", height=460, xaxis_ticksuffix="%",
                          margin=dict(l=10, r=40, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        f_sorted = tbl.sort_values("순자금(5일 추정)")
        fig = go.Figure(go.Bar(
            x=f_sorted["순자금(5일 추정)"] / 1e6, y=f_sorted["섹터"], orientation="h",
            marker_color=[UP if v >= 0 else DOWN for v in f_sorted["순자금(5일 추정)"]],
            text=[fmt_usd(v) for v in f_sorted["순자금(5일 추정)"]], textposition="outside",
            customdata=np.stack([f_sorted["상대거래량"], f_sorted["CMF(20)"]], axis=-1),
            hovertemplate="%{y}<br>순자금 %{text}<br>상대거래량 %{customdata[0]:.2f}x"
                          "<br>CMF %{customdata[1]:+.3f}<extra></extra>"))
        fig.update_layout(title="섹터 ETF 순자금 유입 추정 (최근 5거래일, $M)", height=460,
                          margin=dict(l=10, r=40, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns([3, 2])
    with c3:
        heat = tbl.set_index("섹터")[list(PERIODS)]
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=heat.columns, y=heat.index, colorscale="RdYlGn", zmid=0,
            text=np.vectorize(lambda v: f"{v:+.1f}%")(heat.values), texttemplate="%{text}",
            colorbar=dict(ticksuffix="%")))
        fig.update_layout(title="기간별 섹터 성과 히트맵", height=460, margin=dict(l=10, r=10, t=50, b=10),
                          yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    with c4:
        fig = go.Figure(go.Scatter(
            x=tbl["선택기간"], y=tbl["CMF(20)"], mode="markers+text", text=tbl["섹터"],
            textposition="top center",
            marker=dict(size=(tbl["상대거래량"].fillna(1) * 22).clip(8, 60),
                        color=tbl["선택기간"], colorscale="RdYlGn", cmid=0, line=dict(width=1, color="#333"))))
        fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
        fig.add_vline(x=0, line_dash="dot", line_color=NEUTRAL)
        fig.update_layout(title="수익률 vs 매수압력(CMF) · 원 크기 = 상대거래량", height=460,
                          xaxis_title=f"등락률 ({p_label}, %)", yaxis_title="CMF(20)",
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("우상단 = 가격↑·매수세↑(강세 확인) · 좌하단 = 가격↓·매도세↑(약세 확인) · "
                   "엇갈리면 다이버전스(추세 전환 가능성) 신호로 볼 수 있어요.")

    with st.expander("📋 전체 표 보기 / 자금흐름 지표 설명"):
        show = tbl.drop(columns=["선택기간"]).copy()
        show["순자금(5일 추정)"] = show["순자금(5일 추정)"].map(fmt_usd)
        st.dataframe(show.style.format({**{k: "{:+.2f}%" for k in PERIODS},
                                        "상대거래량": "{:.2f}x", "CMF(20)": "{:+.3f}"}),
                     width="stretch", hide_index=True)
        st.markdown(
            "- **순자금(5일 추정)**: 매일 *(종가가 고가~저가 중 어디서 끝났는지 −1~+1) × 거래대금*을 5일 합산. "
            "실제 ETF 설정/환매(fund flow) 데이터는 유료라서, 무료 시세로 만든 **추정치**입니다.\n"
            "- **상대거래량**: 최근 5일 평균 거래량 ÷ 직전 50일 평균. 1.0x 이상이면 평소보다 관심 증가.\n"
            "- **CMF(20)**: Chaikin Money Flow. +0.05 이상 매수 우위, −0.05 이하 매도 우위.")

# ═════════════════════════════════════════════
# 3) 강세·약세 이유 (뉴스)
# ═════════════════════════════════════════════
with tab_news:
    p_label = st.session_state.get("sector_period", "1주")
    tbl_n = sector_table(PERIODS[p_label])
    top_n = st.slider("강세/약세 섹터 개수", 1, 5, 3)
    st.caption(f"‘섹터 & 자금흐름’ 탭의 기간(**{p_label}**) 기준으로 상위·하위 섹터를 고르고, "
               "해당 ETF와 대표 종목의 최신 뉴스를 모았습니다. 😊/😟 표시는 헤드라인 키워드로 추정한 것이에요.")

    strong, weak = tbl_n.head(top_n), tbl_n.tail(top_n).iloc[::-1]

    def render_sector_news(row, color):
        name, t, chg = row["섹터"], row["티커"], row["선택기간"]
        holdings = SECTORS[t][1]
        news = load_news(tuple([t] + holdings))
        score = news["sentiment"].mean() if not news.empty else 0
        mood = "😊 긍정 우세" if score > 0.15 else "😟 부정 우세" if score < -0.15 else "😐 혼재"
        st.markdown(f"#### <span style='color:{color}'>{name} ({t}) {chg:+.2f}%</span>", unsafe_allow_html=True)
        st.caption(f"뉴스 톤: {mood} · 순자금 {fmt_usd(row['순자금(5일 추정)'])} · "
                   f"상대거래량 {row['상대거래량']:.2f}x · 대표종목 {', '.join(holdings)}")
        if news.empty:
            st.info("가져온 뉴스가 없습니다.")
            return
        for _, n in news.head(6).iterrows():
            icon = "🟢" if n["sentiment"] > 0 else "🔴" if n["sentiment"] < 0 else "⚪"
            link = f"[{n['title']}]({n['url']})" if n["url"] else n["title"]
            st.markdown(f"{icon} {link}  \n<small>`{n['ticker']}` · {n['provider']} · {ago(n['time'])}</small>",
                        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🚀 강세 섹터")
        with st.spinner("뉴스 수집 중…"):
            for _, r in strong.iterrows():
                with st.container(border=True):
                    render_sector_news(r, UP)
    with c2:
        st.subheader("🔻 약세 섹터")
        with st.spinner("뉴스 수집 중…"):
            for _, r in weak.iterrows():
                with st.container(border=True):
                    render_sector_news(r, DOWN)

# ═════════════════════════════════════════════
# 4) 기술적 분석 (기본 COIN)
# ═════════════════════════════════════════════
with tab_ta:
    with st.spinner(f"{ta_ticker} 데이터 불러오는 중…"):
        # 200일선 계산을 위해 넉넉히 받아오고, 화면에는 선택 기간만 표시
        fetch_period = {"3mo": "1y", "6mo": "2y", "1y": "2y", "2y": "5y"}[ta_period]
        full = load_prices((ta_ticker,), fetch_period).get(ta_ticker)

    if full is None or len(full) < 60:
        st.error(f"'{ta_ticker}' 데이터를 가져오지 못했습니다. 티커를 확인해 주세요.")
    else:
        full = A.add_indicators(full)
        n_show = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504}[ta_period]
        d = full.tail(n_show)
        vp = A.volume_profile(d, bins=vp_bins)
        sr = A.support_resistance(d, order=swing_order)
        hvn = A.high_volume_nodes(vp["levels"])
        last = d.iloc[-1]
        prev = d.iloc[-2]

        m = st.columns(5)
        m[0].metric(f"{ta_ticker} 종가", f"${last['Close']:,.2f}", f"{(last['Close'] / prev['Close'] - 1) * 100:+.2f}%")
        m[1].metric("1주", f"{A.pct_change_over(full['Close'], 5):+.2f}%")
        m[2].metric("1개월", f"{A.pct_change_over(full['Close'], 21):+.2f}%")
        m[3].metric("RSI(14)", f"{last['RSI']:.0f}")
        m[4].metric("POC(최대 매물대)", f"${vp['poc']:,.2f}",
                    f"{(last['Close'] / vp['poc'] - 1) * 100:+.1f}% (현재가 기준)", delta_color="off")

        fig = make_subplots(
            rows=3, cols=2, shared_xaxes=True, column_widths=[0.82, 0.18], row_heights=[0.62, 0.18, 0.20],
            horizontal_spacing=0.005, vertical_spacing=0.03,
            specs=[[{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None]],
            subplot_titles=(f"{ta_ticker} 일봉 · 이동평균 · 지지/저항", "매물대", "RSI(14)", "MACD"))

        # 캔들 + 이동평균
        fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                     name="가격", increasing_line_color=UP, decreasing_line_color=DOWN),
                      row=1, col=1)
        for ma, color in (("MA20", "#ffb300"), ("MA50", "#42a5f5"), ("MA200", "#ab47bc")):
            fig.add_trace(go.Scatter(x=d.index, y=d[ma], name=ma, line=dict(width=1.3, color=color)), row=1, col=1)

        # Value Area 음영 + POC
        fig.add_hrect(y0=vp["val"], y1=vp["vah"], fillcolor="#90caf9", opacity=0.08, line_width=0, row=1, col=1)
        fig.add_hline(y=vp["poc"], line=dict(color="#ff9800", width=2, dash="dash"), row=1, col=1,
                      annotation_text=f"POC ${vp['poc']:,.2f}", annotation_position="top left",
                      annotation_font_color="#ff9800")

        # 지지 / 저항
        for _, r in sr.iterrows():
            color = UP if r["kind"] == "지지" else DOWN
            fig.add_hline(y=r["price"], line=dict(color=color, width=1 + min(r["touches"], 4) * 0.5),
                          opacity=0.85, row=1, col=1,
                          annotation_text=f"{r['kind']} ${r['price']:,.2f} ({r['touches']}회)",
                          annotation_position="bottom right", annotation_font_color=color,
                          annotation_font_size=11)

        # 매물대 (가로 막대)
        lv = vp["levels"]
        in_va = (lv["price_mid"] >= vp["val"]) & (lv["price_mid"] <= vp["vah"])
        fig.add_trace(go.Bar(x=lv["volume"], y=lv["price_mid"], orientation="h", name="매물대",
                             width=(lv["price_high"] - lv["price_low"]) * 0.9,
                             marker_color=np.where(in_va, "rgba(66,165,245,0.75)", "rgba(158,158,158,0.45)"),
                             hovertemplate="$%{y:,.2f}<br>거래량 %{x:,.0f}<extra></extra>"),
                      row=1, col=2)
        fig.update_yaxes(matches="y", showticklabels=False, row=1, col=2)
        fig.update_xaxes(showticklabels=False, row=1, col=2)

        # RSI
        fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI", line=dict(color="#7e57c2", width=1.4)),
                      row=2, col=1)
        for lvl, c in ((70, DOWN), (30, UP)):
            fig.add_hline(y=lvl, line=dict(color=c, dash="dot", width=1), row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

        # MACD
        fig.add_trace(go.Bar(x=d.index, y=d["hist"], name="MACD 히스토그램",
                             marker_color=np.where(d["hist"] >= 0, UP, DOWN)), row=3, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["macd"], name="MACD", line=dict(color="#29b6f6", width=1.3)),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d["signal"], name="Signal", line=dict(color="#ffa726", width=1.3)),
                      row=3, col=1)

        fig.update_layout(height=900, xaxis_rangeslider_visible=False, hovermode="x unified", bargap=0,
                          legend=dict(orientation="h", y=1.04, x=0), margin=dict(l=10, r=10, t=60, b=10))
        for row in (1, 2, 3):
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=row, col=1)
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns([3, 2])
        with c1:
            with st.container(border=True):
                st.subheader("🧭 간단 분석 요약")
                for line in A.technical_summary(d, sr, vp):
                    st.markdown(f"- {line}")
                st.caption("규칙 기반 자동 요약이며 매수·매도 권유가 아닙니다.")
        with c2:
            with st.container(border=True):
                st.subheader("📍 레벨 표")
                lv_tbl = sr.assign(
                    거리=lambda x: (x["price"] / last["Close"] - 1) * 100,
                    최근터치=lambda x: x["last_touch"].dt.strftime("%Y-%m-%d"),
                )[["kind", "price", "거리", "touches", "최근터치"]].rename(
                    columns={"kind": "구분", "price": "가격", "touches": "터치"})
                st.dataframe(lv_tbl.sort_values("가격", ascending=False)
                             .style.format({"가격": "${:,.2f}", "거리": "{:+.1f}%"}),
                             width="stretch", hide_index=True)
                st.caption("매물 집중 가격대(HVN): " + " · ".join(f"${p:,.2f}" for p in hvn))

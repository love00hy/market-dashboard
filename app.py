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
import news_ko as N

st.set_page_config(page_title="마켓 대시보드", page_icon="📈", layout="wide")

# ─────────────────────────────────────────────
# 설정값
# ─────────────────────────────────────────────
INDICES = {
    "^GSPC": "S&P 500",
    "^IXIC": "나스닥 종합",
    "^DJI": "다우존스",
    "^KS11": "코스피",
    "BTC-USD": "비트코인",
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
FLOW_WINDOWS = {"5일": 5, "10일": 10, "20일": 20, "60일": 60}
# 봉 주기별 표시 기간(개월)
RANGES = {
    "일봉": {"1개월": 1, "3개월": 3, "6개월": 6, "1년": 12, "2년": 24},
    "주봉": {"6개월": 6, "1년": 12, "2년": 24, "3년": 36, "5년": 60},
    "월봉": {"2년": 24, "3년": 36, "5년": 60, "10년": 120},
}
DEFAULT_RANGE = {"일봉": "6개월", "주봉": "2년", "월봉": "5년"}

UP, DOWN, NEUTRAL = "#26a69a", "#ef5350", "#9e9e9e"


# ─────────────────────────────────────────────
# 데이터 로딩 (캐시)
# ─────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner=False)
def load_prices(tickers: tuple[str, ...], period: str = "1y") -> dict[str, pd.DataFrame]:
    raw = yf.download(list(tickers), period=period, interval="1d", group_by="ticker",
                      auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df.dropna(subset=["Close"])  # 휴장일(다른 종목만 거래된 날) 빈 행 제거
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
    return df.sort_values("time", ascending=False, na_position="last").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def translate_cached(titles: tuple[str, ...]) -> list[str] | None:
    return N.translate_ko(list(titles))


@st.cache_data(ttl=3600, show_spinner=False)
def ai_cached(prompt: str, api_key: str, model: str) -> str | None:
    return N.ai_summary(api_key, prompt, model)


@st.cache_data(ttl=3600, show_spinner=False)
def next_earnings(ticker: str) -> str:
    try:
        cal = yf.Ticker(ticker).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if dates:
            return pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
    except Exception:
        pass
    return "정보 없음"


# ─────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────
def period_change(df: pd.DataFrame, key) -> float:
    return A.ytd_change(df["Close"]) if key == "YTD" else A.pct_change_over(df["Close"], key)


def sym_of(ticker: str) -> str:
    return "₩" if ticker.endswith((".KS", ".KQ")) or ticker == "^KS11" else "$"


def fmt_money(x: float, sym: str = "$") -> str:
    if pd.isna(x):
        return "-"
    sign = "+" if x > 0 else "-" if x < 0 else ""
    x = abs(x)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if x >= div:
            return f"{sign}{sym}{x / div:,.1f}{unit}"
    return f"{sign}{sym}{x:,.0f}"


def signed(x: float, sym: str) -> str:
    return "-" if pd.isna(x) else f"{'+' if x >= 0 else '-'}{sym}{abs(x):,.0f}"


def ago(ts) -> str:
    if pd.isna(ts):
        return ""
    mins = (datetime.now(timezone.utc) - ts.to_pydatetime()).total_seconds() / 60
    if mins < 60:
        return f"{mins:.0f}분 전"
    if mins < 60 * 24:
        return f"{mins / 60:.0f}시간 전"
    return f"{mins / 1440:.0f}일 전"


def slice_months(df: pd.DataFrame, months: int) -> pd.DataFrame:
    return df[df.index >= df.index[-1] - pd.DateOffset(months=months)]


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def news_with_ko(news: pd.DataFrame) -> pd.DataFrame:
    """제목 한국어 번역 컬럼(title_ko) 추가."""
    news = news.copy()
    if news.empty:
        news["title_ko"] = []
        return news
    ko = translate_cached(tuple(news["title"]))
    news["title_ko"] = ko if ko else news["title"]
    return news


def render_headlines(news: pd.DataFrame, show: int = 4):
    def line(n):
        icon = "🟢" if n["sentiment"] > 0 else "🔴" if n["sentiment"] < 0 else "⚪"
        title = n["title_ko"]
        url = n["url"] if isinstance(n["url"], str) and n["url"].startswith("http") else None
        link = f"[{title}]({url})" if url else title
        orig = f" · <i>{n['title']}</i>" if n["title_ko"] != n["title"] else ""
        st.markdown(f"{icon} {link}  \n<small>`{n['ticker']}` · {n['provider']} · {ago(n['time'])}{orig}</small>",
                    unsafe_allow_html=True)
    for _, n in news.head(show).iterrows():
        line(n)
    if len(news) > show:
        with st.expander(f"뉴스 {len(news) - show}건 더 보기"):
            for _, n in news.iloc[show:].iterrows():
                line(n)


def render_tone(news: pd.DataFrame):
    sc = N.tone_score(news)
    color = UP if sc >= 6 else DOWN if sc <= 4 else NEUTRAL
    pos, neg = int((news["sentiment"] > 0).sum()), int((news["sentiment"] < 0).sum())
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;margin:4px 0 8px'>"
        f"<span style='font-size:1.6rem;font-weight:700;color:{color}'>{sc:.1f}<small style='font-size:0.9rem'> / 10</small></span>"
        f"<div style='flex:1'><div style='background:#e0e0e033;border-radius:6px;height:10px'>"
        f"<div style='width:{sc * 10:.0f}%;background:{color};height:10px;border-radius:6px'></div></div>"
        f"<small>뉴스 톤 {N.tone_label(sc)} · 긍정 {pos} / 부정 {neg} / 전체 {len(news)}건</small></div></div>",
        unsafe_allow_html=True)


def render_rule_summary(news: pd.DataFrame):
    rs = N.rule_summary(news)
    if rs["themes"]:
        st.markdown(" ".join(f"`{t}`" for t in rs["themes"]))
    for b in rs["bullets"]:
        st.markdown(f"- {b}")


# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    if st.button("🔄 데이터 새로고침", width="stretch"):
        st.cache_data.clear()
    st.caption("데이터는 10분간 캐시됩니다.")
    st.divider()
    st.subheader("🪙 내 관심 종목")
    ta_ticker = st.text_input("티커 (예: COIN, NVDA, 005930.KS, BTC-USD)", value="COIN").strip().upper() or "COIN"
    st.markdown("**💼 내 포지션** (차트에 평단선 표시)")
    pc1, pc2 = st.columns(2)
    avg_cost = pc1.number_input("평단가", min_value=0.0, value=0.0, step=0.01, format="%.2f",
                                help="보유하지 않았으면 0")
    cur_qty = pc2.number_input("보유 수량", min_value=0, value=0, step=1)
    vp_bins = st.slider("매물대 구간 수", 20, 80, 40, step=5)
    swing_order = st.slider("스윙 감도(작을수록 레벨 많음)", 2, 15, 5)
    st.divider()
    st.subheader("🤖 AI 뉴스 요약 (선택)")
    api_key = get_secret("ANTHROPIC_API_KEY") or st.text_input(
        "Claude API 키", type="password", help="입력하면 뉴스를 한국어로 원인 분석·요약합니다. 없으면 번역+키워드 요약만 제공.")
    ai_model = get_secret("ANTHROPIC_MODEL", "claude-haiku-5-5")
    st.caption("✅ AI 요약 켜짐" if api_key else "번역 + 키워드 요약 모드")
    st.divider()
    st.caption("⚠️ 투자 참고용입니다. 투자 판단과 책임은 본인에게 있습니다.")

SYM = sym_of(ta_ticker)

st.title("📈 마켓 대시보드")
st.caption(f"Yahoo Finance 기준 (약 15분 지연) · 조회 시각 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

tab_idx, tab_sec, tab_news, tab_ta = st.tabs(
    ["📊 주요 지수", "🏭 섹터 & 자금흐름", "📰 강세·약세 이유(뉴스)", f"🪙 {ta_ticker} 기술적 분석"])

# ═════════════════════════════════════════════
# 1) 주요 지수
# ═════════════════════════════════════════════
with tab_idx:
    with st.spinner("지수 데이터 불러오는 중…"):
        idx_data = load_prices(tuple(INDICES) + (ta_ticker,), "10y")

    cols = st.columns(len(INDICES))
    for col, (t, name) in zip(cols, INDICES.items()):
        df = idx_data.get(t)
        if df is None or len(df) < 2:
            col.metric(name, "-")
            continue
        last, prev = df["Close"].iloc[-1], df["Close"].iloc[-2]
        col.metric(name, f"{last:,.2f}", f"{(last / prev - 1) * 100:+.2f}%",
                   delta_color="inverse" if t == "^VIX" else "normal")

    cc = st.columns([2, 1.3, 2.5])
    sel = cc[0].selectbox("비교할 지수", [t for t in INDICES if t != "^VIX"], format_func=lambda t: INDICES[t])
    iv = cc[1].radio("봉", ["일봉", "주봉", "월봉"], horizontal=True, key="idx_iv")
    rng = cc[2].radio("기간", list(RANGES[iv]), index=list(RANGES[iv]).index(DEFAULT_RANGE[iv]),
                      horizontal=True, key=f"idx_rng_{iv}")
    months = RANGES[iv][rng]

    c1, c2 = st.columns([1, 1])
    with c1:
        series = {}
        for t, label in ((sel, INDICES[sel]), (ta_ticker, ta_ticker)):
            if t in idx_data:
                series[label] = slice_months(A.resample_ohlc(idx_data[t], iv), months)["Close"]
        fig = go.Figure()
        if series:
            comb = pd.concat(series, axis=1).sort_index().ffill().dropna()
            norm = (comb / comb.iloc[0] - 1) * 100
            colors = ["#42a5f5", "#ffb300"]
            for (label, s), color in zip(norm.items(), colors):
                fig.add_trace(go.Scatter(x=s.index, y=s, name=label, mode="lines", line=dict(width=2, color=color)))
            if norm.shape[1] == 2:
                gap = norm.iloc[:, 1] - norm.iloc[:, 0]
                fig.add_trace(go.Bar(x=gap.index, y=gap, name=f"{ta_ticker} 초과수익(%p)", opacity=0.25,
                                     marker_color=np.where(gap >= 0, UP, DOWN)))
        fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
        fig.update_layout(title=f"{ta_ticker} vs {INDICES[sel]} 수익률 비교 ({iv} · {rng})", height=460,
                          hovermode="x unified", yaxis_ticksuffix="%", bargap=0,
                          margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.12))
        st.plotly_chart(fig, width="stretch")

        if series and norm.shape[1] == 2:
            r = comb.pct_change().dropna()
            corr = r.iloc[:, 0].corr(r.iloc[:, 1])
            beta = r.iloc[:, 1].cov(r.iloc[:, 0]) / r.iloc[:, 0].var() if r.iloc[:, 0].var() else np.nan
            m = st.columns(4)
            m[0].metric(f"{ta_ticker} 수익률", f"{norm.iloc[-1, 1]:+.2f}%")
            m[1].metric(f"{INDICES[sel]} 수익률", f"{norm.iloc[-1, 0]:+.2f}%")
            m[2].metric("초과수익", f"{norm.iloc[-1, 1] - norm.iloc[-1, 0]:+.2f}%p")
            m[3].metric("상관계수 / 베타", f"{corr:.2f} / {beta:.2f}")
            st.caption("베타 1.5 = 지수가 1% 움직일 때 평균 1.5% 움직임 · 상관계수 1에 가까울수록 지수와 같이 움직임")
        elif ta_ticker not in idx_data:
            st.warning(f"'{ta_ticker}' 데이터를 찾지 못했습니다. 사이드바 티커를 확인해 주세요.")

    with c2:
        if sel in idx_data:
            full = A.resample_ohlc(idx_data[sel], iv)
            full["MA20"], full["MA60"] = full["Close"].rolling(20).mean(), full["Close"].rolling(60).mean()
            d = slice_months(full, months)
            f2 = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"],
                                          close=d["Close"], name=INDICES[sel],
                                          increasing_line_color=UP, decreasing_line_color=DOWN))
            for ma, color in (("MA20", "#ffb300"), ("MA60", "#42a5f5")):
                f2.add_trace(go.Scatter(x=d.index, y=d[ma], name=f"{ma[2:]}{iv[0]} 이평", line=dict(width=1.2, color=color)))
            f2.update_layout(height=460, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=50, b=10),
                             title=f"{INDICES[sel]} {iv} ({rng})", legend=dict(orientation="h", y=-0.12))
            if iv == "일봉" and not sel.endswith("-USD"):  # 코인은 주말에도 거래
                f2.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(f2, width="stretch")

# ═════════════════════════════════════════════
# 2) 섹터 & 자금흐름
# ═════════════════════════════════════════════
with st.spinner("섹터 데이터 불러오는 중…"):
    sec_data = load_prices(tuple(SECTORS), "2y")


def sector_table(period_key, flow_n: int) -> pd.DataFrame:
    rows = []
    for t, (name, _) in SECTORS.items():
        df = sec_data.get(t)
        if df is None:
            continue
        ft = A.flow_trend(df, flow_n)
        rows.append({"티커": t, "섹터": name, "선택기간": period_change(df, period_key),
                     "순유입": ft.get("net_usd", np.nan), "유입강도": ft.get("intensity", np.nan),
                     "직전강도": ft.get("prev", np.nan), "증가율": ft.get("change_pp", np.nan),
                     "기간수익률": ft.get("ret", np.nan), "ADX": ft.get("adx", np.nan),
                     "추세점수": ft.get("score", np.nan), "등급": ft.get("grade", "-"),
                     "코멘트": ft.get("comment", "")})
    return pd.DataFrame(rows).sort_values("선택기간", ascending=False).reset_index(drop=True)


with tab_sec:
    cc = st.columns(2)
    p_label = cc[0].radio("등락률 기간", list(PERIODS), index=1, horizontal=True, key="sec_p")
    f_label = cc[1].radio("자금흐름 기간", list(FLOW_WINDOWS), index=2, horizontal=True, key="sec_f")
    st.session_state["sector_period"] = p_label
    st.session_state["flow_window"] = f_label
    fn = FLOW_WINDOWS[f_label]
    tbl = sector_table(PERIODS[p_label], fn)

    c1, c2 = st.columns(2)
    with c1:
        t_sorted = tbl.sort_values("선택기간")
        fig = go.Figure(go.Bar(
            x=t_sorted["선택기간"], y=t_sorted["섹터"] + " (" + t_sorted["티커"] + ")", orientation="h",
            marker_color=[UP if v >= 0 else DOWN for v in t_sorted["선택기간"]],
            text=[f"{v:+.2f}%" for v in t_sorted["선택기간"]], textposition="outside"))
        fig.update_layout(title=f"섹터별 등락률 ({p_label})", height=440, xaxis_ticksuffix="%",
                          margin=dict(l=10, r=40, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        f_sorted = tbl.sort_values("순유입")
        fig = go.Figure(go.Bar(
            x=f_sorted["순유입"] / 1e6, y=f_sorted["섹터"], orientation="h",
            marker_color=[UP if v >= 0 else DOWN for v in f_sorted["순유입"]],
            text=[fmt_money(v) for v in f_sorted["순유입"]], textposition="outside"))
        fig.update_layout(title=f"순자금 유입 추정 (최근 {f_label}, $M)", height=440,
                          margin=dict(l=10, r=40, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader(f"💧 자금 유입 증가율 ({f_label} 기준)")
    c3, c4 = st.columns(2)
    with c3:
        g = tbl.sort_values("증가율")
        fig = go.Figure(go.Bar(
            x=g["증가율"], y=g["섹터"], orientation="h",
            marker_color=[UP if v >= 0 else DOWN for v in g["증가율"]],
            text=[f"{p:+.1f}% → {c:+.1f}%" for p, c in zip(g["직전강도"], g["유입강도"])], textposition="outside",
            hovertemplate="%{y}<br>증가율 %{x:+.1f}%p<br>직전→최근 유입강도 %{text}<extra></extra>"))
        fig.add_vline(x=0, line_color=NEUTRAL)
        fig.update_layout(title=f"유입 강도 변화 (직전 {f_label} → 최근 {f_label}, %p)", height=460,
                          xaxis_ticksuffix="%p", margin=dict(l=10, r=90, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("유입 강도 = 순유입 추정액 ÷ 총 거래대금. 막대가 오른쪽일수록 직전보다 돈이 더 빠르게 들어오는 중.")
    with c4:
        by_int = tbl.sort_values("유입강도", ascending=False)["섹터"].tolist()
        default = by_int[:2] + by_int[-2:]
        pick = st.multiselect("누적 유입 추이 볼 섹터", tbl["섹터"].tolist(), default=default, key="flow_pick")
        lookback = max(fn * 3, 30)
        fig = go.Figure()
        name2t = {v[0]: k for k, v in SECTORS.items()}
        for nm in pick:
            t = name2t[nm]
            if t in sec_data:
                cf = A.cumulative_flow(sec_data[t], lookback)
                fig.add_trace(go.Scatter(x=cf.index, y=cf, name=nm, mode="lines"))
        if len(sec_data):
            first = next(iter(sec_data.values()))
            cut = first.index[-fn] if len(first) > fn else first.index[0]
            fig.add_vrect(x0=cut, x1=first.index[-1], fillcolor="#90caf9", opacity=0.08, line_width=0,
                          annotation_text=f"최근 {f_label}", annotation_position="top left")
        fig.add_hline(y=0, line_dash="dot", line_color=NEUTRAL)
        fig.update_layout(title=f"누적 순유입 추이 (최근 {lookback}거래일, 평균 일거래대금 대비 %)", height=460,
                          hovermode="x unified", yaxis_ticksuffix="%", margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.15))
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, width="stretch")
        st.caption("선이 꾸준히 오른쪽 위로 가면 지속적인 자금 유입. 100% = 하루치 평균 거래대금만큼 순유입.")

    st.subheader(f"🧭 추세 강도 평가 ({f_label} 기준)")
    c5, c6 = st.columns([2, 3])
    with c5:
        s = tbl.sort_values("추세점수")
        fig = go.Figure(go.Bar(
            x=s["추세점수"], y=s["섹터"], orientation="h",
            marker=dict(color=s["추세점수"], colorscale="RdYlGn", cmin=-100, cmax=100),
            text=[f"{v:+.0f} · {g}" for v, g in zip(s["추세점수"], s["등급"])], textposition="outside"))
        fig.add_vline(x=0, line_color=NEUTRAL)
        fig.update_layout(height=460, xaxis=dict(range=[-130, 130], title="추세 점수 (−100 ~ +100)"),
                          margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")
    with c6:
        view = tbl.sort_values("추세점수", ascending=False)[
            ["섹터", "추세점수", "등급", "기간수익률", "유입강도", "증가율", "ADX", "코멘트"]].rename(
            columns={"기간수익률": f"{f_label} 수익률", "유입강도": "유입강도(%)", "증가율": "증가율(%p)"})
        st.dataframe(view, hide_index=True, width="stretch", height=460, column_config={
            "추세점수": st.column_config.NumberColumn(format="%+d"),
            f"{f_label} 수익률": st.column_config.NumberColumn(format="%+.2f%%"),
            "유입강도(%)": st.column_config.NumberColumn(format="%+.1f"),
            "증가율(%p)": st.column_config.NumberColumn(format="%+.1f"),
            "ADX": st.column_config.NumberColumn(format="%.0f"),
            "코멘트": st.column_config.TextColumn(width="large"),
        })

    with st.expander("ℹ️ 지표 계산 방법"):
        st.markdown(
            "- **순유입 추정**: 매일 *(종가가 당일 고가~저가 중 어디서 끝났는지, −1~+1) × 거래대금*을 합산. "
            "실제 ETF 설정/환매 데이터는 유료라서 무료 시세로 만든 **추정치**입니다.\n"
            "- **유입 강도**: 기간 순유입 ÷ 기간 총 거래대금 × 100. +20% 이상이면 매수세가 강한 편.\n"
            "- **증가율**: 최근 기간 유입강도 − 직전 같은 기간 유입강도 (%p).\n"
            "- **추세 점수**: 가격 모멘텀(변동성 대비 수익률) 40% + 유입 강도 35% + 누적 유입선의 일관성 25%를 "
            "합산하고, ADX(추세 뚜렷함)가 낮으면 점수를 깎습니다. "
            "+45↑ 강한 상승추세 · +15↑ 상승 우위 · ±15 중립 · −15↓ 하락 우위 · −45↓ 강한 하락추세.")

# ═════════════════════════════════════════════
# 3) 강세·약세 이유 (뉴스)
# ═════════════════════════════════════════════
with tab_news:
    p_label = st.session_state.get("sector_period", "1주")
    f_label = st.session_state.get("flow_window", "20일")
    tbl_n = sector_table(PERIODS[p_label], FLOW_WINDOWS[f_label])
    top_n = st.slider("강세/약세 섹터 개수", 1, 5, 3)
    st.caption(f"‘섹터 & 자금흐름’ 탭의 등락률 기간(**{p_label}**) 기준 상위·하위 섹터입니다. "
               + ("Claude가 뉴스를 읽고 원인을 한국어로 요약합니다." if api_key
                  else "헤드라인은 한국어로 번역하고 키워드로 테마를 분류합니다. 사이드바에 Claude API 키를 넣으면 원인 분석 요약이 추가됩니다."))

    strong, weak = tbl_n.head(top_n), tbl_n.tail(top_n).iloc[::-1]

    sector_news = {}
    with st.spinner("뉴스 수집·번역 중…"):
        for _, r in pd.concat([strong, weak]).iterrows():
            t = r["티커"]
            sector_news[t] = news_with_ko(load_news(tuple([t] + SECTORS[t][1])))

    if api_key:
        rows = []
        for _, r in pd.concat([strong, weak]).iterrows():
            heads = "; ".join(sector_news[r["티커"]]["title"].head(4))
            rows.append(f"- {r['섹터']}({r['티커']}) {r['선택기간']:+.2f}%, 순유입 {fmt_money(r['순유입'])}: {heads}")
        with st.container(border=True):
            st.markdown("### 🧾 오늘의 섹터 흐름 한눈에")
            with st.spinner("AI 요약 중…"):
                st.markdown(ai_cached(N.market_prompt(rows, p_label), api_key, ai_model) or "요약 실패")

    def render_sector(row, color):
        t, name, chg = row["티커"], row["섹터"], row["선택기간"]
        news = sector_news[t]
        st.markdown(f"#### <span style='color:{color}'>{name} ({t}) {chg:+.2f}%</span>", unsafe_allow_html=True)
        st.caption(f"순유입 {fmt_money(row['순유입'])} · 추세 {row['등급']} ({row['추세점수']:+.0f}) · "
                   f"대표종목 {', '.join(SECTORS[t][1])}")
        if news.empty:
            st.info("가져온 뉴스가 없습니다.")
            return
        render_tone(news)
        if api_key:
            flow_text = f"최근 {f_label} 순유입 {fmt_money(row['순유입'])}, 추세 '{row['등급']}'"
            with st.spinner("AI 요약 중…"):
                st.markdown(ai_cached(N.sector_prompt(name, t, p_label, chg, flow_text, news), api_key, ai_model)
                            or "")
        else:
            render_rule_summary(news)
        st.markdown("**📰 주요 헤드라인**")
        render_headlines(news, 4)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🚀 강세 섹터")
        for _, r in strong.iterrows():
            with st.container(border=True):
                render_sector(r, UP)
    with c2:
        st.subheader("🔻 약세 섹터")
        for _, r in weak.iterrows():
            with st.container(border=True):
                render_sector(r, DOWN)

# ═════════════════════════════════════════════
# 4) 기술적 분석
# ═════════════════════════════════════════════
with tab_ta:
    base = idx_data.get(ta_ticker)
    if base is None or len(base) < 60:
        st.error(f"'{ta_ticker}' 데이터를 가져오지 못했습니다. 티커를 확인해 주세요. (한국 주식은 005930.KS 형식)")
        st.stop()

    cc = st.columns([1.3, 3])
    t_iv = cc[0].radio("봉", ["일봉", "주봉", "월봉"], horizontal=True, key="ta_iv")
    t_ranges = {"일봉": {"3개월": 3, "6개월": 6, "1년": 12, "2년": 24},
                "주봉": {"1년": 12, "2년": 24, "3년": 36, "5년": 60},
                "월봉": {"3년": 36, "5년": 60, "10년": 120}}[t_iv]
    t_rng = cc[1].radio("기간", list(t_ranges), index=min(2, len(t_ranges) - 1), horizontal=True, key=f"ta_rng_{t_iv}")
    bar_word = {"일봉": "하루", "주봉": "한 주", "월봉": "한 달"}[t_iv]

    full = A.add_indicators(A.resample_ohlc(base, t_iv))
    d = slice_months(full, t_ranges[t_rng])
    if len(d) < 15:
        st.warning("표시할 봉이 너무 적습니다. 기간을 늘려 주세요.")
        st.stop()
    vp = A.volume_profile(d, bins=vp_bins)
    # 표시 봉 수가 많을수록(예: 일봉 2년) 스윙 감도와 묶음 폭을 키워 '큰 레벨' 위주로
    scale = max(1.0, len(d) / 126)
    eff_order = int(round(swing_order * scale))
    eff_tol = 0.6 * float(np.sqrt(scale))
    sr = A.support_resistance(d, order=eff_order, tolerance_atr=eff_tol)
    hi_idx, lo_idx = d["High"].idxmax(), d["Low"].idxmin()
    p_high, p_low = float(d["High"].max()), float(d["Low"].min())
    extra_lv = [(p_high, f"기간 고점 ({hi_idx:%Y-%m-%d})", 0), (p_low, f"기간 저점 ({lo_idx:%Y-%m-%d})", 0)]
    hvn = A.high_volume_nodes(vp["levels"])
    last, prev = d.iloc[-1], d.iloc[-2]
    close, atr_v = float(last["Close"]), float(last["ATR"])

    m = st.columns(5)
    m[0].metric(f"{ta_ticker} 종가", f"{SYM}{close:,.2f}", f"{(close / base['Close'].iloc[-2] - 1) * 100:+.2f}% (전일)")
    m[1].metric("1주", f"{A.pct_change_over(base['Close'], 5):+.2f}%")
    m[2].metric("1개월", f"{A.pct_change_over(base['Close'], 21):+.2f}%")
    m[3].metric(f"RSI(14, {t_iv})", f"{last['RSI']:.0f}")
    m[4].metric("POC(최대 매물대)", f"{SYM}{vp['poc']:,.2f}",
                f"현재가 대비 {(close / vp['poc'] - 1) * 100:+.1f}%", delta_color="off")

    fig = make_subplots(
        rows=3, cols=2, shared_xaxes=True, column_widths=[0.82, 0.18], row_heights=[0.62, 0.18, 0.20],
        horizontal_spacing=0.005, vertical_spacing=0.03,
        specs=[[{}, {}], [{"colspan": 2}, None], [{"colspan": 2}, None]],
        subplot_titles=(f"{ta_ticker} {t_iv} · 이동평균 · 지지/저항", "매물대", "RSI(14)", "MACD"))
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
                                 name="가격", increasing_line_color=UP, decreasing_line_color=DOWN), row=1, col=1)
    for ma, color in (("MA20", "#ffb300"), ("MA50", "#42a5f5"), ("MA200", "#ab47bc")):
        fig.add_trace(go.Scatter(x=d.index, y=d[ma], name=f"{ma[2:]}{t_iv[0]} 이평", line=dict(width=1.3, color=color)),
                      row=1, col=1)
    fig.add_hrect(y0=vp["val"], y1=vp["vah"], fillcolor="#90caf9", opacity=0.08, line_width=0, row=1, col=1)
    fig.add_hline(y=vp["poc"], line=dict(color="#ff9800", width=2, dash="dash"), row=1, col=1,
                  annotation_text=f"POC {SYM}{vp['poc']:,.2f}", annotation_position="top left",
                  annotation_font_color="#ff9800")
    for _, r in sr.iterrows():
        color = UP if r["kind"] == "지지" else DOWN
        fig.add_hline(y=r["price"], line=dict(color=color, width=1 + min(r["touches"], 4) * 0.5), opacity=0.85,
                      row=1, col=1, annotation_text=f"{r['kind']} {SYM}{r['price']:,.2f} ({r['touches']}회)",
                      annotation_position="bottom right", annotation_font_color=color, annotation_font_size=11)
    for y, lab, colr, pos in ((p_high, f"기간 고점 {SYM}{p_high:,.2f} ({hi_idx:%y.%m.%d})", "#e040fb", "top left"),
                              (p_low, f"기간 저점 {SYM}{p_low:,.2f} ({lo_idx:%y.%m.%d})", "#00bcd4", "bottom left")):
        fig.add_hline(y=y, line=dict(color=colr, width=1.5, dash="dashdot"), row=1, col=1,
                      annotation_text=lab, annotation_position=pos, annotation_font_color=colr)
    if avg_cost > 0:
        fig.add_hline(y=avg_cost, line=dict(color="#2962ff", width=3), row=1, col=1,
                      annotation_text=f"내 평단 {SYM}{avg_cost:,.2f}" + (f" ({(close / avg_cost - 1) * 100:+.1f}%)"),
                      annotation_position="top right", annotation_font=dict(color="#2962ff", size=13))
    lv = vp["levels"]
    in_va = (lv["price_mid"] >= vp["val"]) & (lv["price_mid"] <= vp["vah"])
    fig.add_trace(go.Bar(x=lv["volume"], y=lv["price_mid"], orientation="h", name="매물대",
                         width=(lv["price_high"] - lv["price_low"]) * 0.9,
                         marker_color=np.where(in_va, "rgba(66,165,245,0.75)", "rgba(158,158,158,0.45)"),
                         hovertemplate=f"{SYM}%{{y:,.2f}}<br>거래량 %{{x:,.0f}}<extra></extra>"), row=1, col=2)
    fig.update_yaxes(matches="y", showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI"], name="RSI", line=dict(color="#7e57c2", width=1.4)), row=2, col=1)
    for lvl, c in ((70, DOWN), (30, UP)):
        fig.add_hline(y=lvl, line=dict(color=c, dash="dot", width=1), row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["hist"], name="MACD 히스토그램",
                         marker_color=np.where(d["hist"] >= 0, UP, DOWN)), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["macd"], name="MACD", line=dict(color="#29b6f6", width=1.3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["signal"], name="Signal", line=dict(color="#ffa726", width=1.3)), row=3, col=1)
    fig.update_layout(height=900, xaxis_rangeslider_visible=False, hovermode="x unified", bargap=0,
                      legend=dict(orientation="h", y=1.09, x=0), margin=dict(l=10, r=10, t=100, b=10))
    is_crypto = ta_ticker.endswith("-USD")
    if t_iv == "일봉" and not is_crypto:
        for row in (1, 2, 3):
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], row=row, col=1)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"지지/저항 자동 감도: 스윙 {eff_order}봉 · 표시 {len(d)}봉 기준 (기간이 길수록 큰 레벨만 남김)")

    c1, c2 = st.columns([3, 2])
    with c1:
        with st.container(border=True):
            st.subheader("🧭 기술적 분석 요약")
            for line in A.technical_summary(d, sr, vp, sym=SYM, bar=bar_word):
                st.markdown(f"- {line}")

            st.markdown("---")
            st.subheader(f"🗞️ {ta_ticker} 주요 이슈")
            earn = next_earnings(ta_ticker)
            st.caption(f"📅 다음 실적발표 예정일: **{earn}**")
            with st.spinner("뉴스 수집·번역 중…"):
                tnews = news_with_ko(load_news((ta_ticker,), per_ticker=15))
            if tnews.empty:
                st.info("가져온 뉴스가 없습니다.")
            else:
                render_tone(tnews)
                if api_key:
                    with st.spinner("AI 요약 중…"):
                        st.markdown(ai_cached(N.ticker_prompt(ta_ticker, A.pct_change_over(base["Close"], 21),
                                                              tnews, earn), api_key, ai_model) or "")
                else:
                    render_rule_summary(tnews)
                st.markdown("**📰 최신 헤드라인**")
                render_headlines(tnews, 5)
            st.caption("규칙 기반 자동 분석이며 매수·매도 권유가 아닙니다.")
    with c2:
        with st.container(border=True):
            st.subheader("📍 레벨 표")
            lv_tbl = sr.assign(
                거리=lambda x: (x["price"] / close - 1) * 100,
                최근터치=lambda x: x["last_touch"].dt.strftime("%Y-%m-%d"),
            )[["kind", "price", "거리", "touches", "최근터치"]].rename(
                columns={"kind": "구분", "price": "가격", "touches": "터치"})
            st.dataframe(lv_tbl.sort_values("가격", ascending=False)
                         .style.format({"가격": SYM + "{:,.2f}", "거리": "{:+.1f}%"}),
                         width="stretch", hide_index=True)
            st.caption("매물 집중 가격대(HVN): " + " · ".join(f"{SYM}{p:,.2f}" for p in hvn))

    # ─────────── 분할매수 / 분할매도 계산기 ───────────
    st.markdown("---")
    st.subheader("🧮 분할매수 · 분할매도 계획 계산기")
    st.caption(f"현재 {t_iv} 차트의 지지·저항선, 매물대(POC/VA), 매물 집중대를 기준으로 가격을 잡습니다. "
               "지지/저항이 부족하면 ATR(평균 변동폭) 간격으로 보충합니다.")
    is_krw = SYM == "₩"
    i1, i3 = st.columns([2, 1])
    i1.info(f"💼 내 포지션: 평단 {SYM}{avg_cost:,.2f} × {cur_qty:,}주 — 왼쪽 사이드바에서 수정"
            if avg_cost > 0 or cur_qty > 0 else "💼 왼쪽 사이드바 '내 포지션'에 평단가와 보유 수량을 넣으면 반영됩니다.")
    budget = i3.number_input(f"추가 매수 예산 ({SYM})", min_value=0.0, value=10_000_000.0 if is_krw else 10_000.0,
                             step=100_000.0 if is_krw else 100.0, format="%.0f")
    j1, j2, j3, j4 = st.columns(4)
    n_buy = j1.number_input("분할매수 횟수", 1, 10, 5)
    buy_mode = j2.selectbox("매수 배분", ["균등", "뒤로 갈수록 많이", "앞쪽에 많이"],
                            help="'뒤로 갈수록 많이' = 낮은 가격일수록 더 많이 (물타기형)")
    n_sell = j3.number_input("분할매도 횟수", 1, 10, 3)
    sell_mode = j4.selectbox("매도 배분", ["균등", "앞쪽에 많이", "뒤로 갈수록 많이"],
                             help="'앞쪽에 많이' = 가까운 저항에서 먼저 많이 익절")
    k1, k2, k3 = st.columns(3)
    sell_pct = k1.slider("매도할 비중 (%)", 10, 100, 100, step=10)
    min_profit = k2.slider("최소 목표 수익률 (평단 대비 %)", 0, 30, 3,
                           help="1차 매도가가 평단 × (1 + 이 값) 이상이 되도록 그 위의 저항대만 사용")
    include_buys = k3.checkbox("매도 수량에 '매수 계획 수량'까지 포함", value=False,
                               help="끄면 지금 보유 수량·평단 기준으로 매도 계획")

    sr_wide = A.support_resistance(d, order=max(2, eff_order // 2), tolerance_atr=eff_tol, max_levels=25)
    hvn_wide = A.high_volume_nodes(vp["levels"], 10)
    buys, stop = A.plan_buys(close, atr_v, sr_wide, vp, hvn_wide, int(n_buy), budget, buy_mode, cur_qty, avg_cost,
                             extra=extra_lv)
    if include_buys and not buys.empty and buys["누적수량"].iloc[-1] > 0:
        base_qty, base_avg = buys["누적수량"].iloc[-1], buys["매수 후 평단"].iloc[-1]
    else:
        base_qty, base_avg = cur_qty, avg_cost
    sells = A.plan_sells(close, atr_v, sr_wide, vp, hvn_wide, int(n_sell), int(base_qty * sell_pct / 100),
                         base_avg, sell_mode, min_profit=min_profit, extra=extra_lv)

    pfx = SYM
    num = "{:,.0f}" if is_krw else "{:,.2f}"
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("##### 🟢 분할매수 계획")
        st.dataframe(buys.drop(columns=["기준레벨"]).style.format({
            "매수가": pfx + num, "현재가 대비": "{:+.1f}%", "수량": "{:,.0f}", "금액": pfx + "{:,.0f}",
            "누적수량": "{:,.0f}", "매수 후 평단": pfx + num}, na_rep="-"), hide_index=True, width="stretch")
        spent = buys["금액"].sum() if not buys.empty else 0
        st.caption(f"총 매수금액 {pfx}{spent:,.0f} / 예산 {pfx}{budget:,.0f} · "
                   f"모두 체결 시 평단 {pfx}{num.format(buys['매수 후 평단'].iloc[-1]) if not buys.empty and buys['누적수량'].iloc[-1] else '-'}")
    with b2:
        st.markdown("##### 🔴 분할매도 계획")
        if sells.empty or sells["수량"].sum() == 0:
            st.info("매도할 수량이 없습니다. 사이드바에 보유 수량을 입력하거나 '매수 계획 수량 포함'을 켜 주세요.")
        elif base_avg > 0:
            st.caption(f"평단 {pfx}{num.format(base_avg)} × (1+{min_profit}%) = "
                       f"{pfx}{num.format(max(close, base_avg * (1 + min_profit / 100)))} 이상의 저항대에서만 매도가를 잡았습니다.")
        st.dataframe(sells.drop(columns=["기준레벨"]).style.format({
            "매도가": pfx + num, "현재가 대비": "{:+.1f}%", "수량": "{:,.0f}", "금액": pfx + "{:,.0f}",
            "수익률": "{:+.1f}%", "실현손익": lambda v: signed(v, pfx)}, na_rep="-"), hide_index=True, width="stretch")
        if not sells.empty and sells["실현손익"].notna().any():
            st.caption(f"전량 체결 시 예상 실현손익 {signed(sells['실현손익'].sum(), pfx)} "
                       f"(기준 평단 {pfx}{num.format(base_avg)}, 매도 {int(sells['수량'].sum()):,}주)")

    s1, s2, s3 = st.columns(3)
    s1.metric("손절 참고가", f"{pfx}{num.format(stop)}" if pd.notna(stop) else "-",
              f"{(stop / close - 1) * 100:+.1f}% (마지막 매수 레벨 −ATR, 최대 10%)" if pd.notna(stop) else None, delta_color="off")
    if not buys.empty and buys["누적수량"].iloc[-1] > 0 and pd.notna(stop):
        loss = (stop - buys["매수 후 평단"].iloc[-1]) * buys["누적수량"].iloc[-1]
        s2.metric("손절 시 예상 손익", signed(loss, pfx))
    if not sells.empty and len(buys):
        avg_buy = (buys["금액"].sum() / buys["수량"].sum()) if buys["수량"].sum() else np.nan
        avg_sell = (sells["금액"].sum() / sells["수량"].sum()) if sells["수량"].sum() else np.nan
        if pd.notna(avg_buy) and pd.notna(avg_sell) and pd.notna(stop):
            rr = (avg_sell - avg_buy) / (avg_buy - stop) if avg_buy > stop else np.nan
            s3.metric("손익비 (평균 매도 vs 손절)", f"{rr:.2f} : 1" if pd.notna(rr) else "-")

    # 계획 차트
    pdx = d.tail(150)
    pf = go.Figure(go.Candlestick(x=pdx.index, open=pdx["Open"], high=pdx["High"], low=pdx["Low"], close=pdx["Close"],
                                  name=ta_ticker, increasing_line_color=UP, decreasing_line_color=DOWN))
    for _, r in buys.iterrows():
        pf.add_hline(y=r["매수가"], line=dict(color=UP, dash="dash", width=1.5),
                     annotation_text=f"{r['차수']} 매수 {pfx}{num.format(r['매수가'])} ×{r['수량']:,}",
                     annotation_position="bottom left", annotation_font_color=UP)
    for _, r in sells.iterrows():
        pf.add_hline(y=r["매도가"], line=dict(color=DOWN, dash="dash", width=1.5),
                     annotation_text=f"{r['차수']} 매도 {pfx}{num.format(r['매도가'])} ×{r['수량']:,}",
                     annotation_position="top left", annotation_font_color=DOWN)
    if avg_cost > 0:
        pf.add_hline(y=avg_cost, line=dict(color="#2962ff", width=3),
                     annotation_text=f"내 평단 {pfx}{num.format(avg_cost)}", annotation_position="top right",
                     annotation_font=dict(color="#2962ff", size=13))
    if pd.notna(stop):
        pf.add_hline(y=stop, line=dict(color="#ff9800", width=2),
                     annotation_text=f"손절 참고 {pfx}{num.format(stop)}", annotation_position="bottom right",
                     annotation_font_color="#ff9800")
    pf.update_layout(title="분할매수·매도 계획 차트", height=560, xaxis_rangeslider_visible=False, showlegend=False,
                     margin=dict(l=10, r=10, t=50, b=10))
    if t_iv == "일봉" and not is_crypto:
        pf.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(pf, width="stretch")
    st.caption("매수가는 지지선 살짝 위(+0.1 ATR), 매도가는 저항선 살짝 아래(−0.1 ATR)로 잡아 체결 가능성을 높였습니다. "
               "계산 결과는 참고용입니다.")

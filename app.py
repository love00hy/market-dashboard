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
import analysts as AN
import fundamentals as FU
import gurus as G
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


def tone_score(news: pd.DataFrame) -> float:
    """뉴스 톤 점수 0~10 (5 = 중립). 최신 뉴스일수록 가중치↑."""
    if news.empty:
        return 5.0
    s = news["sentiment"].astype(float).values
    w = 0.85 ** np.arange(len(s))
    return round(5 + 5 * float((s * w).sum() / w.sum()), 1)


def tone_label(score: float) -> str:
    if score >= 7.5:
        return "😄 매우 긍정"
    if score >= 6:
        return "🙂 긍정"
    if score > 4:
        return "😐 중립"
    if score > 2.5:
        return "🙁 부정"
    return "😣 매우 부정"


def render_tone(news: pd.DataFrame):
    sc = tone_score(news)
    color = UP if sc >= 6 else DOWN if sc <= 4 else NEUTRAL
    pos, neg = int((news["sentiment"] > 0).sum()), int((news["sentiment"] < 0).sum())
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;margin:4px 0 8px'>"
        f"<span style='font-size:1.6rem;font-weight:700;color:{color}'>{sc:.1f}<small style='font-size:0.9rem'> / 10</small></span>"
        f"<div style='flex:1'><div style='background:#e0e0e033;border-radius:6px;height:10px'>"
        f"<div style='width:{sc * 10:.0f}%;background:{color};height:10px;border-radius:6px'></div></div>"
        f"<small>뉴스 톤 {tone_label(sc)} · 긍정 {pos} / 부정 {neg} / 전체 {len(news)}건</small></div></div>",
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
    ta_ticker = st.text_input("티커 (예: COIN, NVDA, 005930.KS, BTC-USD)", value="",
                              placeholder="티커를 입력하세요").strip().upper()
    st.markdown("**💼 내 포지션** (차트에 평단선 표시)")
    pc1, pc2 = st.columns(2)
    avg_cost = pc1.number_input("평단가", min_value=0.0, value=0.0, step=0.01, format="%.2f",
                                help="보유하지 않았으면 0")
    cur_qty = pc2.number_input("보유 수량", min_value=0, value=0, step=1)
    vp_bins = st.slider("매물대 구간 수", 20, 80, 40, step=5)
    swing_order = st.slider("스윙 감도(작을수록 레벨 많음)", 2, 15, 5)
    st.divider()
    st.subheader("🤖 AI 해설 (선택)")
    api_key = get_secret("ANTHROPIC_API_KEY") or st.text_input(
        "Claude API 키", type="password", help="입력하면 거장 매매·애널리스트 의견·종목 이슈를 AI가 한국어로 해설합니다. 없어도 자동 요약은 제공.")
    ai_model = get_secret("ANTHROPIC_MODEL", "claude-haiku-5-5")
    st.caption("✅ AI 해설 켜짐" if api_key else "자동 요약 모드 (AI 해설 꺼짐)")
    st.divider()
    st.caption("⚠️ 투자 참고용입니다. 투자 판단과 책임은 본인에게 있습니다.")

SYM = sym_of(ta_ticker)

st.title("📈 마켓 대시보드")
st.caption(f"Yahoo Finance 기준 (약 15분 지연) · 조회 시각 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

tab_idx, tab_guru, tab_overlap, tab_analyst, tab_fund, tab_ta = st.tabs(
    ["📊 주요 지수", "🏛️ 거장 포트폴리오", "🔁 거장 중복 종목", "📑 애널리스트 강세·약세", "🧮 재무 비교",
     f"🪙 {ta_ticker or '종목'} 기술적 분석"])

# ═════════════════════════════════════════════
# 1) 주요 지수
# ═════════════════════════════════════════════
with tab_idx:
    with st.spinner("지수 데이터 불러오는 중…"):
        idx_data = load_prices(tuple(INDICES) + ((ta_ticker,) if ta_ticker else ()), "10y")

    _pos_df = idx_data.get(ta_ticker)
    if _pos_df is not None and avg_cost > 0 and cur_qty > 0:
        _px = float(_pos_df["Close"].iloc[-1])
        _pl = (_px - avg_cost) * cur_qty
        _pr = (_px / avg_cost - 1) * 100
        _c = UP if _pl >= 0 else DOWN
        st.sidebar.markdown(
            f"<div style='border:1px solid {_c};border-radius:8px;padding:8px 10px;margin-top:-6px'>"
            f"<small>{ta_ticker} 종가 {SYM}{_px:,.2f} 기준</small><br>"
            f"<b style='color:{_c};font-size:1.25rem'>{'+' if _pl >= 0 else '-'}{SYM}{abs(_pl):,.0f} ({_pr:+.2f}%)</b><br>"
            f"<small>평가금액 {SYM}{_px * cur_qty:,.0f} · 매입금액 {SYM}{avg_cost * cur_qty:,.0f}</small></div>",
            unsafe_allow_html=True)

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
            if t and t in idx_data:
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
        fig.update_layout(title=(f"{ta_ticker} vs {INDICES[sel]}" if ta_ticker else INDICES[sel]) + f" 수익률 ({iv} · {rng})", height=460,
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
        elif not ta_ticker:
            st.info("👈 사이드바에 관심 종목 티커를 입력하면 지수와 수익률을 비교해 드립니다.")
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
# 2) 거장 포트폴리오 (SEC 13F)
# ═════════════════════════════════════════════
SEC_UA = get_secret("SEC_USER_AGENT", "MarketDashboard personal-research contact@marketdashboard.app")
FIGI_KEY = get_secret("OPENFIGI_API_KEY", "")
STATUS_ICON = {"신규": "🆕 신규", "증가": "🟢 증가", "유지": "⚪ 유지", "감소": "🟠 감소", "청산": "🔴 청산"}
STATUS_COLOR = {"신규": "#00c853", "증가": UP, "유지": "#90a4ae", "감소": "#ffa726", "청산": DOWN}


@st.cache_resource
def _http():
    import requests
    return requests.Session()


@st.cache_resource
def _figi_store() -> dict:
    return {}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def guru_data(cik: int, ua: str) -> dict:
    try:
        return G.load_guru(_http(), cik, ua)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def tickers_for(cusips: list[str]) -> dict:
    store = _figi_store()
    try:
        G.map_cusips(_http(), cusips, store, FIGI_KEY)
    except Exception:
        pass
    return store


def label_of(cusip: str, name: str, tk: dict) -> str:
    t = tk.get(cusip)
    return f"{t} · {str(name).title()}" if t else str(name).title()


with tab_guru:
    sel_gurus = st.multiselect("👤 거장·기관 선택", list(G.GURUS), default=G.DEFAULT_GURUS,
                               format_func=lambda g: f"{g} ({G.GURUS[g][1]})")
    st.caption("출처: SEC EDGAR 13F-HR 공시 (분기 말 기준, 분기 종료 후 최대 45일 뒤 공개) · 미국 상장 주식 롱 포지션만 포함, "
               "공매도·현금·해외주식·옵션 제외")
    guru_res: dict[str, dict] = {}
    if sel_gurus:
        prog = st.progress(0.0, text="SEC 공시 불러오는 중…")
        for i, g in enumerate(sel_gurus, 1):
            guru_res[g] = guru_data(G.GURUS[g][0], SEC_UA)
            prog.progress(i / len(sel_gurus), text=f"SEC 공시 불러오는 중… {g}")
        prog.empty()
    ok = {g: r for g, r in guru_res.items() if "holdings" in r}
    for g, r in guru_res.items():
        if "error" in r:
            st.warning(f"{g}: 공시를 불러오지 못했습니다 ({r['error']})")

    need = []
    for r in ok.values():
        h = r["holdings"]
        need += h.head(25)["cusip"].tolist() + h[h["status"] == "청산"].head(5)["cusip"].tolist()
    with st.spinner("종목 티커 변환 중…"):
        TK = tickers_for(need) if need else {}

    if ok:
        ov_rows = []
        for g, r in ok.items():
            h = r["holdings"]
            ov_rows.append({"거장": g, "운용사": G.GURUS[g][1], "기준 분기": r["period"], "공시일": r["filed"],
                            "평가액": r["total"], "전분기 대비": r["value_chg"], "종목 수": int((h["shares"] > 0).sum()),
                            "신규": int((h["status"] == "신규").sum()), "증가": int((h["status"] == "증가").sum()),
                            "감소": int((h["status"] == "감소").sum()), "청산": int((h["status"] == "청산").sum()),
                            "최대 보유": label_of(h.iloc[0]["cusip"], h.iloc[0]["name"], TK) if len(h) else "-"})
        ovdf = pd.DataFrame(ov_rows)
        c1, c2 = st.columns([2, 3])
        with c1:
            s = ovdf.sort_values("전분기 대비")
            fig = go.Figure(go.Bar(x=s["전분기 대비"], y=s["거장"], orientation="h",
                                   marker_color=[UP if v >= 0 else DOWN for v in s["전분기 대비"].fillna(0)],
                                   text=[f"{v:+.1f}%" for v in s["전분기 대비"]], textposition="outside"))
            fig.update_layout(title="포트폴리오 평가액 증감률 (전분기 대비)", height=380, xaxis_ticksuffix="%",
                              margin=dict(l=10, r=40, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("##### 한눈에 보기")
            st.dataframe(ovdf.drop(columns=["운용사"]).style.format(
                {"평가액": lambda v: fmt_money(v).lstrip("+"), "전분기 대비": "{:+.1f}%"}, na_rep="-"),
                hide_index=True, width="stretch", height=380)

        st.markdown("---")
        pick = st.selectbox("🔎 자세히 볼 거장", list(ok), format_func=lambda g: f"{g} ({G.GURUS[g][1]})")
        r = ok[pick]
        h = r["holdings"].copy()
        h["종목"] = [label_of(c, n, TK) for c, n in zip(h["cusip"], h["name"])]
        m = st.columns(5)
        m[0].metric("기준 분기", r["period"], f"공시 {r['filed']}", delta_color="off")
        m[1].metric("13F 평가액", fmt_money(r["total"]).lstrip("+"), f"{r['value_chg']:+.1f}% (전분기)")
        m[2].metric("보유 종목", f"{int((h['shares'] > 0).sum())}개")
        m[3].metric("상위 5종목 비중", f"{h[h['shares'] > 0].head(5)['weight'].sum():.0f}%")
        m[4].metric("신규 / 청산", f"{int((h['status'] == '신규').sum())} / {int((h['status'] == '청산').sum())}")

        with st.container(border=True):
            st.markdown(f"**💬 {pick} 이번 분기 코멘트** <small>(공시 데이터 기반 자동 요약 · 본인 발언 아님)</small>",
                        unsafe_allow_html=True)
            for line in G.guru_comment(r, TK):
                st.markdown(f"- {line}")
            if api_key:
                with st.spinner("AI 해설 작성 중…"):
                    st.markdown("**🤖 AI 해설**  \n" + (ai_cached(G.guru_ai_prompt(pick, r, TK), api_key, ai_model) or ""))

        c1, c2 = st.columns([3, 2])
        with c1:
            top = h[h["shares"] > 0].head(15).iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top["weight"], y=top["종목"], orientation="h",
                marker_color=[STATUS_COLOR[s] for s in top["status"]],
                text=[f"{w:.1f}% · {STATUS_ICON[s]}" + (f" {c:+.0f}%" if s in ("증가", "감소") else "")
                      for w, s, c in zip(top["weight"], top["status"], top["share_chg"])],
                textposition="outside"))
            fig.update_layout(title=f"{pick} 상위 15 보유종목 (비중 %, 색 = 이번 분기 매매)", height=560,
                              xaxis=dict(ticksuffix="%", range=[0, top["weight"].max() * 1.45 if len(top) else 1]),
                              margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            buys = h[h["status"].isin(["신규", "증가"])].sort_values("weight", ascending=False).head(8)
            sells = h[h["status"].isin(["감소", "청산"])].sort_values("weight_prev", ascending=False).head(8)
            st.markdown("##### 🟢 이번 분기 매수")
            st.dataframe(pd.DataFrame({"종목": buys["종목"], "구분": buys["status"].map(STATUS_ICON),
                                       "주식수 변화": buys["share_chg"], "현재 비중": buys["weight"]})
                         .style.format({"주식수 변화": "{:+.0f}%", "현재 비중": "{:.2f}%"}, na_rep="신규"),
                         hide_index=True, width="stretch")
            st.markdown("##### 🔴 이번 분기 매도")
            st.dataframe(pd.DataFrame({"종목": sells["종목"], "구분": sells["status"].map(STATUS_ICON),
                                       "주식수 변화": sells["share_chg"], "직전 비중": sells["weight_prev"]})
                         .style.format({"주식수 변화": "{:+.0f}%", "직전 비중": "{:.2f}%"}, na_rep="-"),
                         hide_index=True, width="stretch")

        with st.expander(f"📋 {pick} 전체 보유내역 ({len(h)}줄)"):
            full = pd.DataFrame({"종목": h["종목"], "상태": h["status"].map(STATUS_ICON), "비중": h["weight"],
                                 "직전 비중": h["weight_prev"], "평가액": h["value"], "주식수": h["shares"],
                                 "주식수 변화": h["share_chg"], "CUSIP": h["cusip"]})
            st.dataframe(full.style.format({"비중": "{:.2f}%", "직전 비중": "{:.2f}%",
                                            "평가액": lambda v: fmt_money(v).lstrip("+"),
                                            "주식수": "{:,.0f}", "주식수 변화": "{:+.1f}%"}, na_rep="-"),
                         hide_index=True, width="stretch")
            st.caption("티커는 OpenFIGI로 CUSIP을 변환한 것이며, 변환 안 된 종목은 회사명으로 표시됩니다.")
    elif sel_gurus:
        st.error("선택한 거장의 공시를 하나도 불러오지 못했습니다. 잠시 후 '데이터 새로고침'을 눌러 주세요.")

# ═════════════════════════════════════════════
# 3) 거장 중복 종목
# ═════════════════════════════════════════════
with tab_overlap:
    ov = G.overlap(ok) if ok else pd.DataFrame()
    if ov.empty:
        st.info("‘거장 포트폴리오’ 탭에서 거장을 2명 이상 선택해 주세요.")
    else:
        TK = tickers_for(ov.head(60)["cusip"].tolist())
        ov["티커"] = ov["cusip"].map(TK)
        ov["종목"] = [label_of(c, n, TK) for c, n in zip(ov["cusip"], ov["name"])]
        cc = st.columns([2, 2, 3])
        min_h = cc[0].slider("최소 보유 거장 수", 1, max(2, len(ok)), min(2, len(ok)))
        top_k = cc[1].slider("표시 개수", 10, 50, 20, step=5)
        cc[2].caption(f"선택한 거장 {len(ok)}명 기준 · 점수 = 보유 거장 수×10 + 이번 분기 매수 거장×5 + 신규 매수×3 "
                      "− 매도 거장×4 + 평균 비중×0.5")
        view = ov[ov["holders"] >= min_h].head(top_k)

        c1, c2 = st.columns([3, 2])
        with c1:
            s = view.iloc[::-1]
            fig = go.Figure(go.Bar(
                x=s["score"], y=s["종목"], orientation="h",
                marker=dict(color=s["holders"], colorscale="Tealgrn", showscale=True,
                            colorbar=dict(title="보유<br>거장 수")),
                text=[f"{int(hh)}명 보유 · 매수 {int(b)} / 매도 {int(sl)}" for hh, b, sl in
                      zip(s["holders"], s["buyers"], s["sellers"])],
                textposition="outside", customdata=s["gurus"],
                hovertemplate="%{y}<br>점수 %{x}<br>%{customdata}<extra></extra>"))
            fig.update_layout(title="거장 중복 보유 종목 점수", height=max(420, 26 * len(s) + 80),
                              xaxis=dict(range=[0, s["score"].max() * 1.5 if len(s) else 1]),
                              margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")
        with c2:
            with st.container(border=True):
                st.markdown("**💬 중복 종목 코멘트** <small>(자동 요약)</small>", unsafe_allow_html=True)
                for _, rr in view.head(6).iterrows():
                    trend = ("이번 분기 **매수세**" if rr["buyers"] > rr["sellers"] else
                             "이번 분기 **매도세**" if rr["sellers"] > rr["buyers"] else "보유 유지")
                    st.markdown(f"- **{rr['종목']}** — 거장 {int(rr['holders'])}명 보유, {trend} "
                                f"(매수 {int(rr['buyers'])} · 매도 {int(rr['sellers'])}, 평균 비중 {rr['avg_weight']:.1f}%)")
                if api_key and len(view):
                    body = "\n".join(f"- {x['종목']}: 보유 {int(x['holders'])}명 [{x['gurus']}], 매수 {int(x['buyers'])}, "
                                     f"매도 {int(x['sellers'])}" for _, x in view.head(12).iterrows())
                    prompt = ("너는 한국 개인투자자를 돕는 애널리스트다. 아래는 유명 투자자들의 13F 공시에서 여러 명이 "
                              f"동시에 보유한 종목 목록이다.\n\n{body}\n\n이 데이터만 근거로 한국어 3~4문장으로: 거장들이 "
                              "공통으로 선호하는 테마/섹터, 이번 분기 매수가 몰린 종목, 주의할 점(13F는 45일 지연된 과거 데이터). "
                              "투자자들의 발언을 지어내지 마라.")
                    with st.spinner("AI 해설 작성 중…"):
                        st.markdown("**🤖 AI 해설**  \n" + (ai_cached(prompt, api_key, ai_model) or ""))
            newbuys = ov[ov["new"] >= 1].sort_values(["new", "holders"], ascending=False).head(10)
            st.markdown("##### 🆕 이번 분기 새로 산 종목 (신규 매수 거장 수 순)")
            st.dataframe(pd.DataFrame({"종목": newbuys["종목"], "신규 매수": newbuys["new"].astype(int),
                                       "보유 거장": newbuys["holders"].astype(int)}),
                         hide_index=True, width="stretch")

        st.markdown("##### 📋 중복 보유 종목 표")
        st.dataframe(pd.DataFrame({
            "종목": view["종목"], "점수": view["score"], "보유 거장 수": view["holders"].astype(int),
            "보유 거장 (비중)": view["gurus"], "이번 분기 매수": view["buyers"].astype(int),
            "신규": view["new"].astype(int), "매도": view["sellers"].astype(int), "평균 비중": view["avg_weight"],
        }).style.format({"점수": "{:.1f}", "평균 비중": "{:.2f}%"}), hide_index=True, width="stretch")
        st.caption("👈 관심 가는 종목은 사이드바 '내 관심 종목'에 티커를 넣으면 기술적 분석 탭에서 바로 볼 수 있어요.")

# ═════════════════════════════════════════════
# 4) 애널리스트 강세·약세
# ═════════════════════════════════════════════


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def analyst_bulk(tickers: tuple[str, ...]) -> dict[str, tuple]:
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        tk = yf.Ticker(t)
        out = [None, None, None]
        for i, attr in enumerate(("upgrades_downgrades", "analyst_price_targets", "recommendations")):
            try:
                out[i] = getattr(tk, attr)
            except Exception:
                pass
        return t, (AN.tidy_actions(out[0]), out[1] if isinstance(out[1], dict) else {},
                   out[2] if isinstance(out[2], pd.DataFrame) else None)
    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(ex.map(one, tickers))


with tab_analyst:
    cc = st.columns([1.3, 1.3, 1.6, 2.5])
    days = {"30일": 30, "60일": 60, "90일": 90}[cc[0].radio("의견 집계 기간", ["30일", "60일", "90일"], index=2,
                                                             horizontal=True)]
    uni = cc[1].radio("종목 범위", ["핵심 30", "확장 60"], horizontal=True)
    add_guru = cc[2].checkbox("거장 중복 상위 15종목 포함", value=True)
    extra = cc[3].text_input("추가로 볼 티커 (쉼표로)", placeholder="예: SOFI, IONQ, RKLB")
    universe = list(AN.UNIVERSE_CORE) + (list(AN.UNIVERSE_EXT) if uni == "확장 60" else [])
    if add_guru and ok:
        ovx = G.overlap(ok)
        if not ovx.empty:
            tkx = tickers_for(ovx.head(30)["cusip"].tolist())
            universe += [tkx[c] for c in ovx[ovx["holders"] >= 2].head(15)["cusip"] if tkx.get(c)]
    universe += [x.strip().upper() for x in extra.split(",") if x.strip()]
    if ta_ticker:
        universe.append(ta_ticker)
    universe = tuple(dict.fromkeys(universe))

    with st.spinner(f"증권사 의견 수집 중… ({len(universe)}종목, 처음 한 번은 30초~1분 걸려요)"):
        raw = analyst_bulk(universe)
    sums = [AN.summarize(t, *raw[t], days=days) for t in universe if t in raw]
    adf = pd.DataFrame([{k: v for k, v in s_.items() if k != "actions"} for s_ in sums])
    SM = {s_["ticker"]: s_ for s_ in sums}
    if adf.empty or adf["firms"].sum() == 0:
        st.error("애널리스트 데이터를 가져오지 못했습니다. 잠시 후 '데이터 새로고침'을 눌러 주세요.")
    else:
        n_show = st.slider("강세/약세 표시 개수", 5, 20, 10)
        strong = adf.sort_values(["bull_score", "bull_firms"], ascending=False).head(n_show)
        # 약세: 매도·하향 근거가 있고, 약세 점수가 강세 점수의 절반 이상인 종목만
        weak = adf[(adf["bear_score"] >= 15) & (adf["bear_score"] >= adf["bull_score"] * 0.5)] \
            .sort_values(["bear_score", "bear_firms"], ascending=False).head(n_show)
        st.caption(f"최근 {days}일 증권사별 **최신 의견**을 집계했습니다. 강세 점수 = 매수 의견 증권사 수(35) + 컨센서스 매수 비율(25) "
                   "+ 목표가 상승여력(20) + 상향·목표가 상향 모멘텀(20). **같은 종목에 매수 의견을 낸 증권사가 많을수록 점수가 높아요.**")

        def score_chart(df_, col, firms_col, title, color):
            s_ = df_.iloc[::-1]
            fig = go.Figure(go.Bar(
                x=s_[col], y=s_["ticker"], orientation="h", marker_color=color,
                text=[f"{v:.0f}점 · {int(f)}곳" + (f" · 여력 {u:+.0f}%" if pd.notna(u) else "")
                      for v, f, u in zip(s_[col], s_[firms_col], s_["upside"])], textposition="outside"))
            fig.update_layout(title=title, height=max(380, 30 * len(s_) + 80), xaxis=dict(range=[0, 135]),
                              margin=dict(l=10, r=10, t=50, b=10))
            return fig

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(score_chart(strong, "bull_score", "bull_firms", "🚀 애널리스트 강세 종목 (점수 · 매수 의견 증권사 수)", UP),
                            width="stretch")
        with c2:
            if weak.empty:
                st.info("집계 기간 내 매도·하향 의견이 뚜렷한 종목이 없습니다.")
            else:
                st.plotly_chart(score_chart(weak, "bear_score", "bear_firms", "🔻 애널리스트 약세 종목 (점수 · 매도 의견 증권사 수)", DOWN),
                                width="stretch")

        def cards(df_, bullish: bool):
            for _, rr in df_.head(5).iterrows():
                sm = SM[rr["ticker"]]
                with st.container(border=True):
                    st.markdown(f"**{rr['ticker']}** · {'강세' if bullish else '약세'} 점수 "
                                f"**{rr['bull_score' if bullish else 'bear_score']:.0f}**")
                    st.markdown(f"<small>💬 {AN.ticker_comment(sm)}</small>", unsafe_allow_html=True)
                    acts = sm["actions"]
                    acts = acts[acts["to"].map(AN.stance) == (1 if bullish else -1)].head(3) if len(acts) else acts
                    for _, a in acts.iterrows():
                        st.markdown(f"<small>· {a['date']:%m/%d} {AN.action_comment(a, sym_of(rr['ticker']))}</small>",
                                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🚀 강세 TOP 5 — 증권사 코멘트 <small>(레이팅·목표가 기반 자동 요약)</small>", unsafe_allow_html=True)
            cards(strong, True)
        with c2:
            st.markdown("##### 🔻 약세 TOP 5 — 증권사 코멘트", unsafe_allow_html=True)
            cards(weak, False)

        with st.expander("📋 전체 종목 표"):
            tbl = adf.sort_values("bull_score", ascending=False)[
                ["ticker", "bull_score", "bear_score", "bull_firms", "bear_firms", "ups", "downs", "raises", "lowers",
                 "buy_ratio", "current", "mean_t", "upside"]].rename(columns={
                    "ticker": "티커", "bull_score": "강세점수", "bear_score": "약세점수", "bull_firms": "매수의견 증권사",
                    "bear_firms": "매도의견 증권사", "ups": "상향", "downs": "하향", "raises": "목표가↑", "lowers": "목표가↓",
                    "buy_ratio": "컨센서스 매수%", "current": "현재가", "mean_t": "평균 목표가", "upside": "상승여력"})
            st.dataframe(tbl.style.format({"강세점수": "{:.0f}", "약세점수": "{:.0f}", "컨센서스 매수%": "{:.0f}%",
                                           "현재가": "{:,.2f}", "평균 목표가": "{:,.2f}", "상승여력": "{:+.1f}%"}, na_rep="-"),
                         hide_index=True, width="stretch")

        # ── 종목 상세
        st.markdown("---")
        opts = list(strong["ticker"]) + [t for t in adf["ticker"] if t not in set(strong["ticker"])]
        dflt = opts.index(ta_ticker) if ta_ticker in opts else 0
        tsel = st.selectbox("🔎 종목별 애널리스트 의견 자세히", opts, index=dflt)
        sm = SM[tsel]
        sy = sym_of(tsel)
        m = st.columns(5)
        m[0].metric("강세 / 약세 점수", f"{sm['bull_score']:.0f} / {sm['bear_score']:.0f}")
        m[1].metric("컨센서스 매수 비율", f"{sm['buy_ratio']:.0f}%" if pd.notna(sm["buy_ratio"]) else "-",
                    f"애널리스트 {sm['n_analysts']}명", delta_color="off")
        m[2].metric("평균 목표가", f"{sy}{sm['mean_t']:,.2f}" if sm["mean_t"] else "-",
                    f"상승여력 {sm['upside']:+.1f}%" if pd.notna(sm["upside"]) else None)
        m[3].metric(f"최근 {days}일 매수 의견", f"{sm['bull_firms']}곳", f"매도 {sm['bear_firms']}곳", delta_color="off")
        m[4].metric("상향 / 하향", f"{sm['ups']} / {sm['downs']}", f"목표가 ↑{sm['raises']} ↓{sm['lowers']}", delta_color="off")

        c1, c2 = st.columns([2, 3])
        with c1:
            if sm["current"] and sm["low_t"] and sm["high_t"]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[sm["low_t"], sm["high_t"]], y=[0, 0], mode="lines",
                                         line=dict(color="#90a4ae", width=10), name="목표가 범위", hoverinfo="skip"))
                pts = [("최저", sm["low_t"], DOWN), ("평균", sm["mean_t"], "#42a5f5"), ("최고", sm["high_t"], UP),
                       ("현재가", sm["current"], "#ffb300")]
                if avg_cost > 0 and tsel == ta_ticker:
                    pts.append(("내 평단", avg_cost, "#2962ff"))
                for nm_, v, col in pts:
                    fig.add_trace(go.Scatter(x=[v], y=[0], mode="markers+text", marker=dict(size=16, color=col,
                                             symbol="diamond" if nm_ in ("현재가", "내 평단") else "circle"),
                                             text=[f"{nm_}<br>{sy}{v:,.0f}"], textposition="top center", name=nm_))
                tg = sm["actions"].dropna(subset=["pt"]).drop_duplicates("firm")
                if len(tg):
                    fig.add_trace(go.Scatter(x=tg["pt"], y=[-0.3] * len(tg), mode="markers", name="증권사별 목표가",
                                             marker=dict(size=9, color="#ab47bc"), text=tg["firm"],
                                             hovertemplate="%{text}: %{x:,.0f}<extra></extra>"))
                fig.update_layout(title=f"{tsel} 목표가 분포", height=300, showlegend=False,
                                  yaxis=dict(visible=False, range=[-0.6, 0.8]), margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, width="stretch")
            with st.container(border=True):
                st.markdown(f"**💬 {tsel} 애널리스트 시각** <small>(자동 요약)</small>", unsafe_allow_html=True)
                st.markdown(AN.ticker_comment(sm))
                with st.spinner("관련 뉴스 수집 중…"):
                    nn = load_news((tsel,), per_ticker=20)
                    nn = nn[nn["title"].map(AN.is_analyst_headline)] if not nn.empty else nn
                    nn = news_with_ko(nn.head(6)) if not nn.empty else nn
                if api_key:
                    with st.spinner("AI 해설 작성 중…"):
                        st.markdown("**🤖 AI 해설**  \n" + (ai_cached(AN.analyst_ai_prompt(
                            sm, nn["title"].tolist() if not nn.empty else []), api_key, ai_model) or ""))
                if not nn.empty:
                    st.markdown("**📰 애널리스트 관련 뉴스**")
                    render_headlines(nn, 6)
        with c2:
            acts = sm["actions"]
            if acts.empty:
                st.info(f"최근 {days}일 내 증권사 의견 변경이 없습니다. 기간을 늘려 보세요.")
            else:
                st.markdown(f"##### 🏦 증권사별 의견 (최근 {days}일, 최신순)")
                st.dataframe(pd.DataFrame({
                    "날짜": acts["date"].dt.strftime("%Y-%m-%d"), "증권사": acts["firm"],
                    "의견": acts["to"].map(lambda g: ("🟢 " if AN.stance(g) > 0 else "🔴 " if AN.stance(g) < 0 else "⚪ ")
                                                     + AN.ko_grade(g)),
                    "변경": acts["action"].map(lambda a: AN.KO_ACTION.get(str(a).lower(), a)),
                    "목표가": acts["pt"], "이전 목표가": acts["pt_prior"],
                    "코멘트": [AN.action_comment(a, sy) for _, a in acts.iterrows()],
                }).style.format({"목표가": sy + "{:,.0f}", "이전 목표가": sy + "{:,.0f}"}, na_rep="-"),
                    hide_index=True, width="stretch", height=min(600, 38 * len(acts) + 40))
        st.caption("데이터: Yahoo Finance 증권사 레이팅·목표가. 코멘트는 레이팅 변경을 요약한 것으로, 증권사 리포트 원문이 아닙니다.")

# ═════════════════════════════════════════════
# 5) 재무 비교 (동종업계 대비 10점 만점)
# ═════════════════════════════════════════════


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def auto_peers(ticker: str) -> tuple[list[str], str]:
    """같은 산업 대표 종목 (Yahoo 산업 분류) → 실패 시 섹터 → 기본 목록."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    for key, cls, label in (("industryKey", "Industry", info.get("industry")),
                            ("sectorKey", "Sector", info.get("sector"))):
        k = info.get(key)
        if not k or not hasattr(yf, cls):
            continue
        try:
            tc = getattr(yf, cls)(k).top_companies
            if tc is not None and len(tc):
                syms = [str(s) for s in tc.index if str(s).upper() != ticker.upper()]
                if syms:
                    return syms[:12], f"{'산업' if cls == 'Industry' else '섹터'}: {label}"
        except Exception:
            pass
    return FU.FALLBACK_PEERS.get(ticker.upper(), []), "기본 경쟁사 목록"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_infos(tickers: tuple[str, ...]) -> dict[str, dict]:
    from concurrent.futures import ThreadPoolExecutor

    def one(t):
        try:
            return t, yf.Ticker(t).info or {}
        except Exception:
            return t, {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(ex.map(one, tickers))


with tab_fund:
    cc = st.columns([1.2, 3])
    f_ticker = cc[0].text_input("분석할 티커", value=ta_ticker, placeholder="예: COIN",
                                key="fund_ticker").strip().upper()
    if not f_ticker:
        st.info("티커를 입력하면 같은 산업 경쟁사와 PER·PBR·ROE·부채비율 등을 비교해 10점 만점으로 점수를 매깁니다.")
    else:
        with st.spinner("경쟁사 찾는 중…"):
            peers_auto, peer_src = auto_peers(f_ticker)
        extra_p = cc[1].text_input("경쟁사 직접 추가 (쉼표로)", placeholder="예: HOOD, SCHW", key="fund_extra")
        extras = [x.strip().upper() for x in extra_p.split(",") if x.strip()]
        options = list(dict.fromkeys(peers_auto + extras))
        peers = st.multiselect(f"비교할 경쟁사 (자동 추천 — {peer_src})", options,
                               default=list(dict.fromkeys(peers_auto[:8] + extras)), key=f"fund_peers_{f_ticker}")
        group = tuple(dict.fromkeys([f_ticker] + peers))
        with st.spinner(f"재무 데이터 불러오는 중… ({len(group)}종목)"):
            infos = fetch_infos(group)
        rows = {t: FU.extract(infos.get(t, {})) for t in group if infos.get(t)}
        if f_ticker not in rows or len(rows) < 3:
            st.warning("비교할 재무 데이터가 부족합니다. 티커를 확인하거나 경쟁사를 2개 이상 선택해 주세요.")
        else:
            fd = FU.score_peers(rows)
            me = fd.loc[f_ticker]
            tot = float(me["total"])
            rank = int((fd["total"].astype(float) > tot).sum() + 1)
            color = UP if tot >= 6.5 else DOWN if tot < 4 else "#ffb300"
            st.markdown(
                f"<div style='display:flex;align-items:baseline;gap:16px;margin:6px 0'>"
                f"<span style='font-size:1.1rem'><b>{f_ticker}</b> {me['name']} · <small>{me['industry']}</small></span></div>"
                f"<div style='display:flex;align-items:center;gap:18px'>"
                f"<span style='font-size:3rem;font-weight:800;color:{color}'>{tot:.1f}<small style='font-size:1.1rem'> / 10</small></span>"
                f"<span>동종 {len(fd)}개 중 <b>{rank}위</b><br><small>지표별 경쟁사 내 순위를 0~10점으로 환산한 평균</small></span></div>",
                unsafe_allow_html=True)
            m = st.columns(4)
            for col_, c in zip(m, FU.CATEGORIES):
                v = me[f"cat_{c}"]
                avg = fd[f"cat_{c}"].astype(float).mean()
                col_.metric(c, f"{v:.1f} / 10" if pd.notna(v) else "-",
                            f"동종 평균 대비 {v - avg:+.1f}" if pd.notna(v) else None)

            c1, c2 = st.columns([2, 3])
            with c1:
                cats = FU.CATEGORIES + [FU.CATEGORIES[0]]
                mine = [float(me[f"cat_{c}"]) if pd.notna(me[f"cat_{c}"]) else 0 for c in FU.CATEGORIES]
                peer_avg = [float(fd[f"cat_{c}"].astype(float).mean()) for c in FU.CATEGORIES]
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=peer_avg + peer_avg[:1], theta=cats, fill="toself", name="동종 평균",
                                              line=dict(color="#90a4ae")))
                fig.add_trace(go.Scatterpolar(r=mine + mine[:1], theta=cats, fill="toself", name=f_ticker,
                                              line=dict(color="#2962ff", width=3)))
                fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10], tickvals=[2, 4, 6, 8, 10])), height=400,
                                  title="분야별 점수 vs 동종 평균", margin=dict(l=30, r=30, t=60, b=20),
                                  legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig, width="stretch")
            with c2:
                s_ = fd.sort_values("total")
                fig = go.Figure(go.Bar(
                    x=s_["total"].astype(float), y=[f"{t} · {n[:18]}" for t, n in zip(s_.index, s_["name"])],
                    orientation="h", marker_color=["#2962ff" if t == f_ticker else "#90a4ae" for t in s_.index],
                    text=[f"{v:.1f}" for v in s_["total"].astype(float)], textposition="outside"))
                fig.update_layout(title="종합 점수 순위 (10점 만점)", height=400, xaxis=dict(range=[0, 11]),
                                  margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, width="stretch")

            with st.container(border=True):
                st.markdown(f"**💬 {f_ticker} 재무 코멘트** <small>(동종 비교 자동 요약)</small>", unsafe_allow_html=True)
                for line in FU.comments(fd, f_ticker):
                    st.markdown(f"- {line}")
                if api_key:
                    with st.spinner("AI 해설 작성 중…"):
                        st.markdown("**🤖 AI 해설**  \n" + (ai_cached(FU.ai_prompt(fd, f_ticker), api_key, ai_model) or ""))

            st.markdown("##### 📋 지표별 비교")
            st.dataframe(FU.metric_table(fd, f_ticker), hide_index=True, width="stretch", height=600,
                         column_config={"점수": st.column_config.ProgressColumn(
                             "점수 (/10)", min_value=0, max_value=10, format="%.1f")})

            with st.expander("🏢 경쟁사 전체 지표 보기"):
                show_k = ["per", "fper", "pbr", "psr", "ev", "roe", "opm", "npm", "rev_g", "eps_g", "de", "cur", "fcf_y"]
                comp = pd.DataFrame({"종목": [f"{t} · {fd.at[t, 'name']}" for t in fd.index],
                                     "종합": fd["total"].astype(float).round(1).values,
                                     **{FU.METRICS[k][0]: [FU.fmt_val(k, fd.at[t, k]) for t in fd.index] for k in show_k}})
                comp = comp.sort_values("종합", ascending=False)
                st.dataframe(comp, hide_index=True, width="stretch")
            st.caption("데이터: Yahoo Finance (최근 12개월·최근 분기 기준). 점수는 '선택한 경쟁사 안에서의 상대 순위'라서 "
                       "경쟁사 구성이 바뀌면 점수도 바뀝니다. 적자 기업의 PER 등은 최하점 처리됩니다.")

# ═════════════════════════════════════════════
# 4) 기술적 분석
# ═════════════════════════════════════════════
with tab_ta:
    if not ta_ticker:
        st.info("👈 왼쪽 사이드바 **'내 관심 종목'**에 티커를 입력하면 기술적 분석, 주요 이슈, "
                "분할매수·매도 계산기가 표시됩니다.\n\n예: `COIN`, `NVDA`, `TSLA`, `005930.KS`(삼성전자), `BTC-USD`(비트코인)")
        st.stop()
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
    sr = A.support_resistance(d, order=eff_order, tolerance_atr=eff_tol, max_levels=8, min_gap_atr=1.0)
    hi_idx, lo_idx = d["High"].idxmax(), d["Low"].idxmin()
    p_high, p_low = float(d["High"].max()), float(d["Low"].min())
    extra_lv = [(p_high, f"기간 고점 ({hi_idx:%Y-%m-%d})", 6.0), (p_low, f"기간 저점 ({lo_idx:%Y-%m-%d})", 6.0)]
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

    if avg_cost > 0 and cur_qty > 0:
        pl = (close - avg_cost) * cur_qty
        pm = st.columns(4)
        pm[0].metric("💼 내 평단", f"{SYM}{avg_cost:,.2f}", f"{cur_qty:,}주 보유", delta_color="off")
        pm[1].metric("평가금액", f"{SYM}{close * cur_qty:,.0f}", f"매입 {SYM}{avg_cost * cur_qty:,.0f}", delta_color="off")
        pm[2].metric("평가손익", signed(pl, SYM), f"{(close / avg_cost - 1) * 100:+.2f}%")
        pm[3].metric("본전까지", f"{(avg_cost / close - 1) * 100:+.2f}%" if close < avg_cost else "수익 구간",
                     f"현재가 {SYM}{close:,.2f}", delta_color="off")

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
    st.caption(f"지지/저항 자동 감도: 스윙 {eff_order}봉 · 표시 {len(d)}봉 기준 — 크게 솟은 고점·저점과 여러 번 닿은 가격 위주로, "
               "서로 4% 이상 떨어진 큰 레벨만 표시합니다.")

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
    q1, q2 = st.columns(2)
    below_avg = q1.checkbox("평단 아래에서만 분할매수 (물타기)", value=True,
                            help="보유 중이면 '현재가'와 '내 평단' 중 낮은 가격 아래의 지지선에서만 매수합니다. 끄면 현재가 아래 지지선 기준(불타기 포함).")
    min_disc = q2.slider("1차 매수는 평단보다 최소 몇 % 아래", 0, 30, 0, disabled=not below_avg)
    j1, j2, j3, j4 = st.columns(4)
    n_buy = j1.number_input("분할매수 횟수", 1, 10, 5)
    buy_mode = j2.selectbox("매수 배분", ["균등", "뒤로 갈수록 많이", "앞쪽에 많이"],
                            help="'뒤로 갈수록 많이' = 낮은 가격일수록 더 많이 (물타기형)")
    n_sell = j3.number_input("분할매도 횟수", 1, 10, 3)
    sell_mode = j4.selectbox("매도 배분", ["균등", "앞쪽에 많이", "뒤로 갈수록 많이"],
                             help="'앞쪽에 많이' = 가까운 저항에서 먼저 많이 익절")
    t1, t2 = st.columns(2)
    target_px = t1.number_input(f"🎯 최종 목표가 ({SYM}, 0 = 자동)", min_value=0.0, value=0.0,
                                step=1.0 if is_krw else 1.0, format="%.0f" if is_krw else "%.2f",
                                help="입력하면 마지막 차수를 이 가격에 두고, 그 사이 큰 저항대에 나머지 차수를 배치합니다.")
    custom_txt = t2.text_input("차수별 매도 비중 직접 입력 (%, 선택)", value="",
                               placeholder="예: 15,15,20,20,30  (분할매도 횟수만큼, 비우면 '매도 배분' 사용)")
    custom_pct = None
    if custom_txt.strip():
        try:
            custom_pct = [float(x) for x in custom_txt.replace(" ", "").split(",") if x]
            if len(custom_pct) != int(n_sell):
                t2.warning(f"분할매도 횟수({int(n_sell)}회)와 개수가 달라 '매도 배분' 방식을 사용합니다.")
                custom_pct = None
        except ValueError:
            t2.warning("숫자와 쉼표만 입력해 주세요.")
            custom_pct = None
    k1, k2, k3 = st.columns(3)
    sell_pct = k1.slider("매도할 비중 (%)", 10, 100, 100, step=10)
    min_profit = k2.slider("최소 목표 수익률 (평단 대비 %)", 0, 30, 3,
                           help="1차 매도가가 평단 × (1 + 이 값) 이상이 되도록 그 위의 저항대만 사용")
    include_buys = k3.checkbox("매도 수량에 '매수 계획 수량'까지 포함", value=False,
                               help="끄면 지금 보유 수량·평단 기준으로 매도 계획")

    sr_wide = A.support_resistance(d, order=max(2, eff_order // 2), tolerance_atr=eff_tol, max_levels=0)
    hvn_wide = A.high_volume_nodes(vp["levels"], 10)
    buys, stop = A.plan_buys(close, atr_v, sr_wide, vp, hvn_wide, int(n_buy), budget, buy_mode, cur_qty, avg_cost,
                             extra=extra_lv, below_avg=below_avg, min_discount=min_disc)
    if include_buys and not buys.empty and buys["누적수량"].iloc[-1] > 0:
        base_qty, base_avg = buys["누적수량"].iloc[-1], buys["매수 후 평단"].iloc[-1]
    else:
        base_qty, base_avg = cur_qty, avg_cost
    sells = A.plan_sells(close, atr_v, sr_wide, vp, hvn_wide, int(n_sell), int(base_qty * sell_pct / 100),
                         base_avg, sell_mode, min_profit=min_profit, extra=extra_lv,
                         target=target_px or None, custom_pct=custom_pct)

    pfx = SYM
    num = "{:,.0f}" if is_krw else "{:,.2f}"
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("##### 🟢 분할매수 계획")
        st.dataframe(buys.drop(columns=["기준레벨"]).style.format({
            "매수가": pfx + num, "현재가 대비": "{:+.1f}%", "평단 대비": "{:+.1f}%", "수량": "{:,.0f}", "금액": pfx + "{:,.0f}",
            "누적수량": "{:,.0f}", "매수 후 평단": pfx + num}, na_rep="-"), hide_index=True, width="stretch")
        spent = buys["금액"].sum() if not buys.empty else 0
        st.caption(f"총 매수금액 {pfx}{spent:,.0f} / 예산 {pfx}{budget:,.0f} · "
                   f"모두 체결 시 평단 {pfx}{num.format(buys['매수 후 평단'].iloc[-1]) if not buys.empty and buys['누적수량'].iloc[-1] else '-'}")
    with b2:
        st.markdown("##### 🔴 분할매도 계획")
        if sells.empty or sells["수량"].sum() == 0:
            st.info("매도할 수량이 없습니다. 사이드바에 보유 수량을 입력하거나 '매수 계획 수량 포함'을 켜 주세요.")
        if target_px and target_px <= max(close, base_avg * (1 + min_profit / 100)):
            st.warning("최종 목표가가 현재가/최소 목표가보다 낮아서 자동 모드로 계산했습니다.")
        if base_avg > 0 and not sells.empty:
            st.caption(f"평단 {pfx}{num.format(base_avg)} × (1+{min_profit}%) = "
                       f"{pfx}{num.format(max(close, base_avg * (1 + min_profit / 100)))} 이상의 저항대에서만 매도가를 잡았습니다.")
        st.dataframe(sells.drop(columns=["기준레벨"]).style.format({
            "매도가": pfx + num, "현재가 대비": "{:+.1f}%", "비중": "{:.0f}%", "수량": "{:,.0f}", "금액": pfx + "{:,.0f}",
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

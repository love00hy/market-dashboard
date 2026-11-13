"""
분석 로직 모음 (데이터 소스와 무관한 순수 함수)
- 수익률 / 자금 유입(프록시) 계산
- 매물대(Volume Profile), POC, Value Area
- 지지·저항(스윙 고점/저점 클러스터링)
- RSI, MACD, 이동평균, ATR
- 뉴스 헤드라인 간단 감성 분류
- 기술적 요약 문장 생성
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# 1. 수익률 / 자금 흐름
# ─────────────────────────────────────────────

def pct_change_over(close: pd.Series, days: int) -> float:
    """최근 종가 대비 N 거래일 전 대비 변화율(%)."""
    close = close.dropna()
    if len(close) <= days:
        return np.nan
    return (close.iloc[-1] / close.iloc[-1 - days] - 1) * 100


def ytd_change(close: pd.Series) -> float:
    close = close.dropna()
    if close.empty:
        return np.nan
    last_date = close.index[-1]
    prev_year = close[close.index.year < last_date.year]
    base = prev_year.iloc[-1] if not prev_year.empty else close.iloc[0]
    return (close.iloc[-1] / base - 1) * 100


def chaikin_money_flow(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """CMF: +면 매수 압력(자금 유입), -면 매도 압력(자금 유출)."""
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl
    mfv = mfm.fillna(0) * df["Volume"]
    return mfv.rolling(n).sum() / df["Volume"].rolling(n).sum()


def money_flow_stats(df: pd.DataFrame, window: int = 5) -> dict:
    """
    ETF의 '자금 유입' 추정치(무료 데이터 기반 프록시).
    - net_flow_usd : 최근 window일 (방향부호 × 거래대금) 합계  → 순매수 대금 추정
    - rel_volume   : 최근 window일 평균 거래량 / 직전 50일 평균
    - cmf          : 20일 Chaikin Money Flow
    """
    df = df.dropna(subset=["Close", "Volume"])
    if len(df) < 60:
        return {"net_flow_usd": np.nan, "rel_volume": np.nan, "cmf": np.nan}
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    # 종가가 당일 범위에서 어디에 위치했는지(-1~+1)를 방향으로 사용
    direction = (((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl).fillna(0)
    dollar_vol = df["Close"] * df["Volume"]
    signed = direction * dollar_vol
    recent_vol = df["Volume"].iloc[-window:].mean()
    base_vol = df["Volume"].iloc[-window - 50:-window].mean()
    return {
        "net_flow_usd": float(signed.iloc[-window:].sum()),
        "rel_volume": float(recent_vol / base_vol) if base_vol else np.nan,
        "cmf": float(chaikin_money_flow(df).iloc[-1]),
    }


# ─────────────────────────────────────────────
# 2. 지표
# ─────────────────────────────────────────────

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["Close"].shift()
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in (20, 50, 200):
        out[f"MA{n}"] = out["Close"].rolling(n).mean()
    out["RSI"] = rsi(out["Close"])
    out = out.join(macd(out["Close"]))
    out["ATR"] = atr(out)
    return out


# ─────────────────────────────────────────────
# 3. 매물대 (Volume Profile)
# ─────────────────────────────────────────────

def volume_profile(df: pd.DataFrame, bins: int = 40, value_area: float = 0.70) -> dict:
    """
    각 봉의 거래량을 그 봉의 고가~저가 구간에 균등 분배해 가격대별 거래량을 계산.
    반환: {'levels': DataFrame(price_low, price_high, price_mid, volume),
           'poc': float, 'vah': float, 'val': float}
    """
    lo, hi = df["Low"].min(), df["High"].max()
    edges = np.linspace(lo, hi, bins + 1)
    vol = np.zeros(bins)
    for h, l, v in zip(df["High"].values, df["Low"].values, df["Volume"].values):
        if not np.isfinite(v) or v <= 0:
            continue
        if h <= l:
            idx = min(np.searchsorted(edges, l, side="right") - 1, bins - 1)
            vol[max(idx, 0)] += v
            continue
        # 봉 범위와 각 bin의 겹치는 길이에 비례해 분배
        overlap = np.clip(np.minimum(edges[1:], h) - np.maximum(edges[:-1], l), 0, None)
        total = overlap.sum()
        if total > 0:
            vol += v * overlap / total

    mids = (edges[:-1] + edges[1:]) / 2
    levels = pd.DataFrame({"price_low": edges[:-1], "price_high": edges[1:],
                           "price_mid": mids, "volume": vol})

    poc_idx = int(np.argmax(vol))
    # Value Area: POC에서 시작해 거래량이 큰 쪽으로 확장, 전체의 70%가 될 때까지
    target = vol.sum() * value_area
    lo_i = hi_i = poc_idx
    acc = vol[poc_idx]
    while acc < target and (lo_i > 0 or hi_i < bins - 1):
        up = vol[hi_i + 1] if hi_i < bins - 1 else -1
        down = vol[lo_i - 1] if lo_i > 0 else -1
        if up >= down:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    return {
        "levels": levels,
        "poc": float(mids[poc_idx]),
        "vah": float(edges[hi_i + 1]),
        "val": float(edges[lo_i]),
    }


def high_volume_nodes(levels: pd.DataFrame, top_n: int = 5) -> list[float]:
    """매물대 상위 가격대(국소 최대) — 지지/저항으로 작동하기 쉬운 구간."""
    v = levels["volume"].values
    peaks = [i for i in range(len(v))
             if (i == 0 or v[i] >= v[i - 1]) and (i == len(v) - 1 or v[i] >= v[i + 1])]
    peaks = sorted(peaks, key=lambda i: v[i], reverse=True)[:top_n]
    return sorted(float(levels["price_mid"].iloc[i]) for i in peaks)


# ─────────────────────────────────────────────
# 4. 지지 / 저항
# ─────────────────────────────────────────────

def swing_points(df: pd.DataFrame, order: int = 5) -> tuple[pd.Series, pd.Series]:
    """좌우 order개 봉보다 높은 고점 / 낮은 저점."""
    hi, lo = df["High"], df["Low"]
    win = 2 * order + 1
    is_high = hi == hi.rolling(win, center=True).max()
    is_low = lo == lo.rolling(win, center=True).min()
    return hi[is_high], lo[is_low]


def support_resistance(df: pd.DataFrame, order: int = 5, tolerance_atr: float = 0.6,
                       max_levels: int = 6) -> pd.DataFrame:
    """
    스윙 고점·저점을 ATR 기반 허용오차로 묶어서 레벨을 만든다.
    많이 터치되고 최근일수록 strength가 높다.
    반환 컬럼: price, touches, last_touch, strength, kind('지지'/'저항')
    """
    highs, lows = swing_points(df, order)
    pts = pd.concat([highs, lows]).sort_values()
    if pts.empty:
        return pd.DataFrame(columns=["price", "touches", "last_touch", "strength", "kind"])

    tol = float(atr(df).iloc[-1]) * tolerance_atr
    clusters: list[list[tuple[pd.Timestamp, float]]] = []
    for ts, price in pts.items():
        if clusters and abs(price - np.mean([p for _, p in clusters[-1]])) <= tol:
            clusters[-1].append((ts, price))
        else:
            clusters.append([(ts, price)])

    last_close = float(df["Close"].iloc[-1])
    n = len(df)
    pos = {ts: i for i, ts in enumerate(df.index)}
    rows = []
    for c in clusters:
        prices = [p for _, p in c]
        last_ts = max(ts for ts, _ in c)
        recency = pos.get(last_ts, 0) / max(n - 1, 1)  # 0(오래됨) ~ 1(최근)
        price = float(np.mean(prices))
        rows.append({
            "price": price,
            "touches": len(c),
            "last_touch": last_ts,
            "strength": len(c) * (0.5 + recency),
            "kind": "지지" if price < last_close else "저항",
        })
    res = pd.DataFrame(rows)
    # 현재가 위/아래에서 각각 가까운 강한 레벨 위주로 선택
    res["dist"] = (res["price"] - last_close).abs() / last_close
    res["score"] = res["strength"] / (1 + 10 * res["dist"])
    res = res.sort_values("score", ascending=False).head(max_levels)
    return res.drop(columns=["dist", "score"]).sort_values("price").reset_index(drop=True)


# ─────────────────────────────────────────────
# 5. 뉴스 감성 (간단 키워드)
# ─────────────────────────────────────────────

_POS = ["beat", "beats", "surge", "soar", "rally", "jump", "gain", "record", "upgrade",
        "raise", "raised", "strong", "growth", "bullish", "outperform", "approval", "approve",
        "rebound", "boost", "tops", "higher", "rise", "rises", "climb", "buy", "deal",
        "inflow", "cut rates", "rate cut", "optimism", "expands", "profit"]
_NEG = ["miss", "misses", "plunge", "tumble", "slump", "drop", "fall", "falls", "sink",
        "downgrade", "cut", "weak", "loss", "bearish", "underperform", "lawsuit", "probe",
        "investigation", "recession", "tariff", "selloff", "sell-off", "lower", "decline",
        "slide", "warn", "warning", "fear", "outflow", "layoff", "ban", "fine", "hack", "crash"]


def headline_sentiment(text: str) -> int:
    """+1 긍정 / 0 중립 / -1 부정. 아주 단순한 키워드 기반 추정."""
    t = f" {text.lower()} "
    pos = sum(1 for w in _POS if f" {w}" in t)
    neg = sum(1 for w in _NEG if f" {w}" in t)
    # 'rate cut' 은 부정 키워드 'cut' 을 상쇄
    if "rate cut" in t or "cut rates" in t:
        neg = max(neg - 1, 0)
    return int(np.sign(pos - neg))


# ─────────────────────────────────────────────
# 6. 기술적 요약
# ─────────────────────────────────────────────

def technical_summary(df: pd.DataFrame, sr: pd.DataFrame, vp: dict) -> list[str]:
    d = df.iloc[-1]
    close = float(d["Close"])
    lines: list[str] = []

    # 추세
    ma = {n: d.get(f"MA{n}") for n in (20, 50, 200)}
    if all(pd.notna(v) for v in ma.values()):
        if close > ma[20] > ma[50] > ma[200]:
            lines.append("📈 정배열(종가 > 20일 > 50일 > 200일) — 강한 상승 추세")
        elif close < ma[20] < ma[50] < ma[200]:
            lines.append("📉 역배열(종가 < 20일 < 50일 < 200일) — 강한 하락 추세")
        else:
            above = [f"{n}일" for n, v in ma.items() if close > v]
            below = [f"{n}일" for n, v in ma.items() if close <= v]
            lines.append(f"↔️ 혼조 — 종가가 {', '.join(above) or '없음'} 선 위, "
                         f"{', '.join(below) or '없음'} 선 아래")

    # RSI
    r = float(d["RSI"])
    if r >= 70:
        lines.append(f"🔥 RSI {r:.0f} — 과매수 구간, 단기 조정 가능성")
    elif r <= 30:
        lines.append(f"🧊 RSI {r:.0f} — 과매도 구간, 기술적 반등 가능성")
    else:
        lines.append(f"RSI {r:.0f} — 중립 구간")

    # MACD
    prev = df.iloc[-2]
    if prev["macd"] <= prev["signal"] and d["macd"] > d["signal"]:
        lines.append("✅ MACD 골든크로스 발생 (최근 봉)")
    elif prev["macd"] >= prev["signal"] and d["macd"] < d["signal"]:
        lines.append("⚠️ MACD 데드크로스 발생 (최근 봉)")
    else:
        lines.append("MACD " + ("시그널 위 — 상승 모멘텀" if d["macd"] > d["signal"]
                                 else "시그널 아래 — 하락 모멘텀"))

    # 매물대
    poc, vah, val = vp["poc"], vp["vah"], vp["val"]
    if close > vah:
        loc = "매물대 상단(VAH) 위 — 매물 소화 후 상방 이탈 상태"
    elif close < val:
        loc = "매물대 하단(VAL) 아래 — 하방 이탈, 위쪽 매물 부담"
    else:
        loc = "주요 매물대(Value Area) 안 — 박스권 성격"
    lines.append(f"🧱 POC ${poc:,.2f} / VA ${val:,.2f}~${vah:,.2f} → {loc}")

    # 가장 가까운 지지/저항
    sup = sr[sr["price"] < close].sort_values("price", ascending=False)
    res = sr[sr["price"] >= close].sort_values("price")
    if not sup.empty:
        s = sup.iloc[0]
        lines.append(f"🟢 1차 지지 ${s.price:,.2f} ({(s.price / close - 1) * 100:+.1f}%, 터치 {s.touches}회)")
    if not res.empty:
        rr = res.iloc[0]
        lines.append(f"🔴 1차 저항 ${rr.price:,.2f} ({(rr.price / close - 1) * 100:+.1f}%, 터치 {rr.touches}회)")
    a = float(d["ATR"])
    lines.append(f"📏 ATR(14) ${a:,.2f} — 하루 평균 변동폭 약 {a / close * 100:.1f}%")
    return lines

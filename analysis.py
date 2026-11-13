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

def technical_summary(df: pd.DataFrame, sr: pd.DataFrame, vp: dict,
                      sym: str = "$", bar: str = "하루") -> list[str]:
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
    lines.append(f"🧱 POC {sym}{poc:,.2f} / VA {sym}{val:,.2f}~{sym}{vah:,.2f} → {loc}")

    # 가장 가까운 지지/저항
    sup = sr[sr["price"] < close].sort_values("price", ascending=False)
    res = sr[sr["price"] >= close].sort_values("price")
    if not sup.empty:
        s = sup.iloc[0]
        lines.append(f"🟢 1차 지지 {sym}{s.price:,.2f} ({(s.price / close - 1) * 100:+.1f}%, 터치 {s.touches}회)")
    if not res.empty:
        rr = res.iloc[0]
        lines.append(f"🔴 1차 저항 {sym}{rr.price:,.2f} ({(rr.price / close - 1) * 100:+.1f}%, 터치 {rr.touches}회)")
    a = float(d["ATR"])
    lines.append(f"📏 ATR(14) {sym}{a:,.2f} — {bar} 평균 변동폭 약 {a / close * 100:.1f}%")
    return lines


# ─────────────────────────────────────────────
# 7. 봉 주기 변환 (일봉 → 주봉/월봉)
# ─────────────────────────────────────────────
INTERVAL_RULES = {"일봉": None, "주봉": "W-FRI", "월봉": "ME"}


def resample_ohlc(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = INTERVAL_RULES.get(interval)
    if rule is None:
        return df
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df[list(agg)].resample(rule).agg(agg).dropna(subset=["Close"])
    # 마지막 봉 날짜를 실제 마지막 거래일로 (진행 중인 주/월 표시용)
    if len(out):
        out.index = list(out.index[:-1]) + [df.index[-1]]
        out.index = pd.DatetimeIndex(out.index)
    return out


# ─────────────────────────────────────────────
# 8. 자금 유입 증가율 & 추세 강도
# ─────────────────────────────────────────────

def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["High"].diff()
    down = -df["Low"].diff()
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    mdm = np.where((down > up) & (down > 0), down, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"adx": dx.ewm(alpha=1 / n, adjust=False).mean(), "pdi": pdi, "mdi": mdi})


def signed_dollar_flow(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(방향부호 × 거래대금, 거래대금). 방향 = 종가의 당일 범위 내 위치(−1~+1)."""
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    direction = (((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / hl).fillna(0)
    dv = df["Close"] * df["Volume"]
    return direction * dv, dv


def cumulative_flow(df: pd.DataFrame, lookback: int) -> pd.Series:
    """최근 lookback일 누적 순유입을 '평균 일거래대금 대비 %'로 표준화 (섹터 간 비교 가능)."""
    sdv, dv = signed_dollar_flow(df.tail(lookback))
    return sdv.cumsum() / dv.mean() * 100


def flow_trend(df: pd.DataFrame, n: int) -> dict:
    """
    n일 기준 자금 흐름 & 추세 강도.
    - intensity    : 최근 n일 순유입 강도(%) = 순유입대금 / 총거래대금
    - prev         : 직전 n일 순유입 강도(%)
    - change_pp    : 증가율(강도 변화, %p)
    - net_usd      : 최근 n일 순유입 추정액
    - ret          : n일 가격 수익률(%)
    - adx          : ADX(14)
    - r2, slope    : 누적 순유입선의 직선성(일관성)과 기울기 방향
    - score        : 추세 점수 −100~+100
    - grade, comment
    """
    df = df.dropna(subset=["Close", "Volume"])
    if len(df) < 2 * n + 15:
        return {}
    sdv, dv = signed_dollar_flow(df)
    cur_int = sdv.iloc[-n:].sum() / dv.iloc[-n:].sum() * 100
    prev_int = sdv.iloc[-2 * n:-n].sum() / dv.iloc[-2 * n:-n].sum() * 100
    ret = (df["Close"].iloc[-1] / df["Close"].iloc[-1 - n] - 1) * 100
    atr_pct = float(atr(df).iloc[-1] / df["Close"].iloc[-1] * 100)
    adx_v = float(adx(df)["adx"].iloc[-1])

    cum = (sdv.iloc[-n:].cumsum() / dv.iloc[-n:].mean()).values
    x = np.arange(len(cum))
    slope, intercept = np.polyfit(x, cum, 1)
    fit = slope * x + intercept
    ss_res, ss_tot = ((cum - fit) ** 2).sum(), ((cum - cum.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    price_s = np.clip(ret / (atr_pct * np.sqrt(n)), -1, 1)
    flow_s = np.clip(cur_int / 20, -1, 1)
    slope_s = np.sign(slope) * max(r2, 0)
    raw = 0.40 * price_s + 0.35 * flow_s + 0.25 * slope_s
    adx_w = float(np.clip((adx_v - 10) / 30, 0.3, 1.0))
    score = int(round(100 * raw * adx_w))

    if score >= 45:
        grade = "강한 상승추세"
    elif score >= 15:
        grade = "상승 우위"
    elif score > -15:
        grade = "중립·횡보"
    elif score > -45:
        grade = "하락 우위"
    else:
        grade = "강한 하락추세"

    notes = []
    if ret > 0 and cur_int < 0:
        notes.append("가격↑ 자금↓ 다이버전스 — 상승 지속력 의심")
    elif ret < 0 and cur_int > 0:
        notes.append("가격↓ 자금↑ — 저가 매집 가능성")
    elif ret > 0 and cur_int > 0:
        notes.append("가격·자금 동반 상승 — " + ("유입 가속 중" if cur_int > prev_int else "유입은 둔화"))
    elif ret < 0 and cur_int < 0:
        notes.append("가격·자금 동반 하락 — " + ("유출 가속 중" if cur_int < prev_int else "유출은 둔화"))
    notes.append(f"ADX {adx_v:.0f} " + ("(추세 뚜렷)" if adx_v >= 25 else "(추세 약함)" if adx_v < 20 else "(추세 형성 중)"))
    return {
        "intensity": float(cur_int), "prev": float(prev_int), "change_pp": float(cur_int - prev_int),
        "net_usd": float(sdv.iloc[-n:].sum()), "ret": float(ret), "adx": adx_v,
        "r2": r2, "slope": float(slope), "score": score, "grade": grade, "comment": " · ".join(notes),
    }


# ─────────────────────────────────────────────
# 9. 분할매수 / 분할매도 계획
# ─────────────────────────────────────────────

def _level_candidates(close: float, sr: pd.DataFrame, vp: dict, hvn: list[float], side: str,
                      extra: list[tuple[float, str, int]] | None = None) -> list[tuple[float, str, int]]:
    """(가격, 근거, 우선순위) — 우선순위 작을수록 중요."""
    c: list[tuple[float, str, int]] = list(extra or [])
    for _, r in sr.iterrows():
        c.append((float(r["price"]), f"{'지지' if r['price'] < close else '저항'}선 (터치 {int(r['touches'])}회)", 0))
    c.append((vp["poc"], "POC (최대 매물대)", 1))
    c.append((vp["val"], "매물대 하단 (VAL)", 2))
    c.append((vp["vah"], "매물대 상단 (VAH)", 2))
    for p in hvn:
        c.append((p, "매물 집중대", 3))
    if side == "buy":
        c = [x for x in c if x[0] < close]
        c.sort(key=lambda x: -x[0])
    else:
        c = [x for x in c if x[0] > close]
        c.sort(key=lambda x: x[0])
    return c


def _pick_levels(cands, n: int, atr_v: float, ref: float, side: str) -> list[tuple[float, str]]:
    """
    ref 기준(매수: 현재가 아래 / 매도: max(현재가, 평단) 위)으로 n개 레벨 선택.
    간격 기준은 ATR과 가격의 %를 함께 써서 월봉처럼 ATR이 큰 경우에도 레벨이 뭉개지지 않게 함.
    """
    near = min(0.3 * atr_v, 0.015 * ref)
    if side == "buy":
        cands = [x for x in cands if x[0] <= ref - near]
    else:
        cands = [x for x in cands if x[0] >= ref + near]
    picked: list[tuple[float, str, int]] = []
    for p, label, pr in cands:
        gap = min(0.5 * atr_v, 0.03 * p)
        clash = [i for i, q in enumerate(picked) if abs(q[0] - p) < gap]
        if clash:  # 너무 가까운 레벨은 하나로 (더 중요한 근거 유지, 근거는 합침)
            i = clash[0]
            q = picked[i]
            if pr < q[2]:
                picked[i] = (p, f"{label} + {q[1]}", pr)
            elif label not in q[1]:
                picked[i] = (q[0], f"{q[1]} + {label}", q[2])
            continue
        picked.append((p, label, pr))
        if len(picked) >= n:
            break
    levels = [(p, l) for p, l, _ in picked]
    # 레벨이 부족하면 변동폭 간격(ATR%, 최대 매수 12% / 매도 20%)으로 보충
    step = min(atr_v / ref, 0.12 if side == "buy" else 0.20)
    step = max(step, 0.03)
    last = levels[-1][0] if levels else ref
    k = 1
    while len(levels) < n:
        last = last * (1 - step) if side == "buy" else last * (1 + step)
        levels.append((last, f"보조 레벨 ({'지지' if side == 'buy' else '저항'} 부재, {step * 100:.0f}% 간격)"))
        k += 1
    return levels[:n]


def _weights(n: int, mode: str) -> np.ndarray:
    if mode == "뒤로 갈수록 많이":
        w = np.linspace(1, 2.5, n)
    elif mode == "앞쪽에 많이":
        w = np.linspace(2.5, 1, n)
    else:
        w = np.ones(n)
    return w / w.sum()


def plan_buys(close: float, atr_v: float, sr, vp, hvn, n: int, budget: float,
              mode: str = "균등", cur_qty: float = 0, avg_cost: float = 0,
              extra=None) -> tuple[pd.DataFrame, float]:
    levels = _pick_levels(_level_candidates(close, sr, vp, hvn, "buy", extra), n, atr_v, close, "buy")
    w = _weights(len(levels), mode)
    rows, qty, cost = [], float(cur_qty), float(cur_qty) * float(avg_cost)
    for i, ((lvl, why), wi) in enumerate(zip(levels, w), 1):
        price = min(lvl + min(0.1 * atr_v, 0.005 * lvl), close)  # 지지선 살짝 위 지정가
        q = int(budget * wi // price) if price > 0 else 0
        qty += q
        cost += q * price
        rows.append({"차수": f"{i}차", "근거": why, "기준레벨": lvl, "매수가": price,
                     "현재가 대비": (price / close - 1) * 100, "수량": q, "금액": q * price,
                     "누적수량": qty, "매수 후 평단": cost / qty if qty else np.nan})
    df = pd.DataFrame(rows)
    # 손절 참고: 마지막 매수 레벨 아래로 ATR(최대 10%)만큼
    stop = levels[-1][0] * (1 - min(atr_v / levels[-1][0], 0.10)) if levels else np.nan
    return df, stop


def plan_sells(close: float, atr_v: float, sr, vp, hvn, m: int, total_qty: float,
               avg_cost: float, mode: str = "균등", min_profit: float = 2.0, extra=None) -> pd.DataFrame:
    """
    매도 레벨은 '현재가'와 '평단 × (1 + 최소수익%)' 중 높은 가격 위의 저항대에서만 고른다
    → 손해 보는 매도가는 나오지 않음.
    """
    floor = close
    if avg_cost > 0:
        floor = max(close, avg_cost * (1 + min_profit / 100))
    cands = _level_candidates(floor, sr, vp, hvn, "sell", extra)
    levels = _pick_levels(cands, m, atr_v, floor, "sell")
    w = _weights(len(levels), mode)
    qtys = np.floor(total_qty * w).astype(int)
    if len(qtys):
        qtys[-1] += int(total_qty) - qtys.sum()  # 나머지는 마지막 차수에
    rows = []
    for i, ((lvl, why), q) in enumerate(zip(levels, qtys), 1):
        price = max(lvl - min(0.1 * atr_v, 0.005 * lvl), floor)  # 저항선 살짝 아래, 단 floor 이상
        pnl = (price - avg_cost) * q if avg_cost > 0 else np.nan
        rows.append({"차수": f"{i}차", "근거": why, "기준레벨": lvl, "매도가": price,
                     "현재가 대비": (price / close - 1) * 100, "수량": int(q), "금액": q * price,
                     "수익률": (price / avg_cost - 1) * 100 if avg_cost > 0 else np.nan, "실현손익": pnl})
    return pd.DataFrame(rows)

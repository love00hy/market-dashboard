"""
종합 매수 후보 점수
거장 중복 · 거장 추정 평단 대비 위치 · 기술적 지표 · 애널리스트 의견 → 각 0~10점 → 가중 합 0~100점
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import analysis as A

COMPONENTS = {"guru": "🏛️ 거장", "cost": "💰 거장 평단", "tech": "📈 기술적", "analyst": "📑 애널리스트"}
DEFAULT_W = {"guru": 30, "cost": 15, "tech": 25, "analyst": 30}


def guru_score(row: pd.Series | None) -> float:
    """overlap 행 → 0~10. 미보유는 0."""
    if row is None:
        return 0.0
    s = (min(row["holders"] / 4, 1) * 6 + min(row["buyers"] / 3, 1) * 3 + (1 if row["new"] > 0 else 0)
         - 1.5 * min(row["sellers"] / 3, 1))
    return float(np.clip(s, 0, 10))


def cost_score(prem: float) -> float:
    """현재가가 거장 평단 대비 몇 % 위인가(prem). 평단 이하 10점, +150% 이상 0점."""
    if prem is None or pd.isna(prem):
        return np.nan
    return float(10 - np.clip(prem, 0, 150) / 150 * 10)


def tech_score(df: pd.DataFrame) -> tuple[float, dict]:
    """일봉 1년 → 0~10점과 근거."""
    if df is None or len(df) < 60:
        return np.nan, {}
    d = A.add_indicators(df)
    last = d.iloc[-1]
    c = float(last["Close"])
    s, why = 0.0, {}
    ma50, ma200 = last.get("MA50"), last.get("MA200")
    if pd.notna(ma50) and c > ma50:
        s += 2
    if pd.notna(ma50) and pd.notna(ma200) and ma50 > ma200:
        s += 2
    if pd.notna(ma200) and c > ma200:
        s += 1
    why["추세"] = ("정배열" if pd.notna(ma200) and c > ma50 > ma200 else
                 "역배열" if pd.notna(ma200) and c < ma50 < ma200 else "혼조")
    r = float(last["RSI"])
    s += 2 if 40 <= r <= 65 else 1.5 if 30 <= r < 40 else 1 if r < 30 or 65 < r <= 75 else 0
    why["RSI"] = r
    if last["macd"] > last["signal"]:
        s += 1.5
    why["MACD"] = "상승" if last["macd"] > last["signal"] else "하락"
    try:
        sr = A.support_resistance(d.tail(250), order=5, max_levels=8)
        sup = sr[sr["price"] < c]["price"]
        dist = (c / sup.max() - 1) * 100 if len(sup) else np.nan
    except Exception:
        dist = np.nan
    if pd.notna(dist):
        s += 1.5 if dist <= 5 else 0.75 if dist <= 10 else 0
    why["지지선까지"] = dist
    return float(np.clip(s, 0, 10)), why


def analyst_score(sm: dict | None) -> float:
    if not sm or (sm.get("firms", 0) == 0 and not sm.get("n_analysts")):
        return np.nan
    return float(np.clip(sm["bull_score"] / 10 - sm["bear_score"] / 20, 0, 10))


def combine(scores: dict[str, float], weights: dict[str, float]) -> float:
    """결측 항목은 빼고 남은 비중으로 다시 나눠 0~100."""
    num = den = 0.0
    for k, w in weights.items():
        v = scores.get(k)
        if v is None or pd.isna(v) or w <= 0:
            continue
        num += v * w
        den += w
    return round(num / den * 10, 1) if den else np.nan


def grade(total: float) -> str:
    if pd.isna(total):
        return "-"
    return ("🟢 강력 매수 후보" if total >= 75 else "🟢 매수 후보" if total >= 60
            else "🟡 관망" if total >= 45 else "🔴 비추천")


def reason(r: dict, sym: str = "$") -> list[str]:
    """종목 카드용 근거 문장."""
    out = []
    if r.get("holders"):
        out.append(f"🏛️ 거장 {int(r['holders'])}명 보유" + (f", 이번 분기 {int(r['buyers'])}명 매수" if r.get("buyers") else "")
                   + (f" ({r['gurus_short']})" if r.get("gurus_short") else ""))
    else:
        out.append("🏛️ 선택한 거장 보유 없음")
    if pd.notna(r.get("prem")):
        out.append(f"💰 현재가가 거장 추정 평단 대비 {r['prem']:+.0f}%"
                   + (" — 거장보다 싸게 살 수 있는 구간" if r["prem"] <= 0 else ""))
    if r.get("trend"):
        rsi = f", RSI {r['rsi']:.0f}" if pd.notna(r.get("rsi")) else ""
        sup = f", 지지선까지 {r['sup_dist']:.1f}%" if pd.notna(r.get("sup_dist")) else ""
        out.append(f"📈 {r['trend']}, MACD {r.get('macd', '-')}{rsi}{sup}")
    if pd.notna(r.get("bull_firms")):
        up = f", 상승여력 {r['upside']:+.0f}%" if pd.notna(r.get("upside")) else ""
        out.append(f"📑 매수 의견 증권사 {int(r['bull_firms'])}곳 / 매도 {int(r.get('bear_firms') or 0)}곳{up}")
    return out


def ai_prompt(top: pd.DataFrame, weights: dict) -> str:
    lines = []
    for _, r in top.iterrows():
        lines.append(f"- {r['ticker']}: 종합 {r['total']:.0f}점 (거장 {r['guru']:.1f}, 평단 {r['cost'] if pd.notna(r['cost']) else '-'}, "
                     f"기술 {r['tech'] if pd.notna(r['tech']) else '-'}, 애널 {r['analyst'] if pd.notna(r['analyst']) else '-'}) | "
                     f"거장 {int(r['holders'])}명, 평단대비 {r['prem'] if pd.notna(r['prem']) else '-'}%, {r['trend']}, "
                     f"RSI {r['rsi'] if pd.notna(r['rsi']) else '-'}, 매수의견 {r['bull_firms']}곳, 상승여력 {r['upside'] if pd.notna(r['upside']) else '-'}%")
    return f"""너는 한국 개인투자자를 돕는 애널리스트다. 아래는 거장 13F 보유, 거장 추정 평단 대비 위치, 기술적 지표,
증권사 의견을 점수화(가중치 {weights})한 매수 후보 상위 종목이다.

{chr(10).join(lines)}

이 데이터만 근거로 한국어 4~5문장으로: 상위 종목들의 공통점, 1~3위가 높은 점수를 받은 이유, 점수가 높아도 주의할 점
(13F는 45일 지연, 추정 평단의 한계 등). 매수를 단정적으로 권유하지 말고 데이터에 없는 내용은 쓰지 마라."""

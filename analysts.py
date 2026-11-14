"""
증권사 애널리스트 분석 (Yahoo Finance 무료 데이터)
- 증권사별 최신 레이팅/목표가 (upgrades_downgrades)
- 컨센서스 (recommendations), 목표가 분포 (analyst_price_targets)
- 강세/약세 점수: '같은 종목에 매수 의견을 낸 증권사가 많을수록' 높은 점수
- 증권사 액션 자동 코멘트
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

BULL = {"buy", "strong buy", "outperform", "overweight", "market outperform", "sector outperform",
        "positive", "accumulate", "add", "conviction buy", "top pick", "long-term buy", "speculative buy",
        "moderate buy", "sector overweight", "outperformer", "strong-buy", "above average"}
BEAR = {"sell", "strong sell", "underperform", "underweight", "market underperform", "sector underperform",
        "negative", "reduce", "moderate sell", "sector underweight", "underperformer"}

KO_GRADE = {
    "buy": "매수", "strong buy": "강력매수", "outperform": "시장수익률 상회", "overweight": "비중확대",
    "market outperform": "시장수익률 상회", "sector outperform": "업종 상회", "positive": "긍정적",
    "accumulate": "매집", "add": "추가매수", "conviction buy": "확신매수", "top pick": "최선호",
    "speculative buy": "투기적 매수", "moderate buy": "다소 매수", "sector overweight": "업종 비중확대",
    "hold": "보유", "neutral": "중립", "equal-weight": "중립(동일비중)", "equal weight": "중립(동일비중)",
    "market perform": "시장수익률", "sector perform": "업종 수준", "in-line": "시장 수준", "peer perform": "동종 수준",
    "sector weight": "업종 비중", "mixed": "혼재", "perform": "시장 수준",
    "sell": "매도", "strong sell": "강력매도", "underperform": "시장수익률 하회", "underweight": "비중축소",
    "market underperform": "시장수익률 하회", "sector underperform": "업종 하회", "negative": "부정적", "reduce": "비중감소",
}
KO_ACTION = {"up": "⬆️ 상향", "down": "⬇️ 하향", "main": "유지", "init": "🆕 신규", "reit": "재확인", "main ": "유지"}


def norm_grade(g) -> str:
    return re.sub(r"\s+", " ", str(g or "").strip().lower())


def stance(g) -> int:
    g = norm_grade(g)
    if g in BULL:
        return 1
    if g in BEAR:
        return -1
    return 0


def ko_grade(g) -> str:
    n = norm_grade(g)
    return KO_GRADE.get(n, str(g) if g else "-")


def tidy_actions(ud: pd.DataFrame | None) -> pd.DataFrame:
    """yfinance upgrades_downgrades → 표준 컬럼 (date, firm, to, from, action, pt, pt_prior, pt_action)."""
    cols = ["date", "firm", "to", "from", "action", "pt", "pt_prior", "pt_action"]
    if ud is None or len(ud) == 0:
        return pd.DataFrame(columns=cols)
    d = ud.reset_index()
    ren = {}
    for c in d.columns:
        lc = str(c).lower().replace(" ", "")
        if lc in ("gradedate", "date", "index"):
            ren[c] = "date"
        elif lc == "firm":
            ren[c] = "firm"
        elif lc == "tograde":
            ren[c] = "to"
        elif lc == "fromgrade":
            ren[c] = "from"
        elif lc == "action":
            ren[c] = "action"
        elif lc == "currentpricetarget":
            ren[c] = "pt"
        elif lc == "priorpricetarget":
            ren[c] = "pt_prior"
        elif lc == "pricetargetaction":
            ren[c] = "pt_action"
    d = d.rename(columns=ren)
    for c in cols:
        if c not in d:
            d[c] = np.nan
    d["date"] = pd.to_datetime(d["date"], errors="coerce", utc=True).dt.tz_localize(None)
    for c in ("pt", "pt_prior"):
        d[c] = pd.to_numeric(d[c], errors="coerce").replace(0, np.nan)
    return d[cols].dropna(subset=["date"]).sort_values("date", ascending=False).reset_index(drop=True)


def pt_direction(r) -> int:
    a = str(r.get("pt_action") or "").lower()
    if a.startswith("rais"):
        return 1
    if a.startswith("lower"):
        return -1
    if pd.notna(r.get("pt")) and pd.notna(r.get("pt_prior")):
        return int(np.sign(r["pt"] - r["pt_prior"]))
    return 0


def summarize(ticker: str, actions: pd.DataFrame, targets: dict | None, recs: pd.DataFrame | None,
              days: int = 90, now: pd.Timestamp | None = None) -> dict:
    """한 종목의 애널리스트 지표 + 강세/약세 점수."""
    now = now or pd.Timestamp.now().normalize()
    a = actions[actions["date"] >= now - pd.Timedelta(days=days)].copy()
    a["stance"] = a["to"].map(stance)
    a["ptdir"] = a.apply(pt_direction, axis=1) if len(a) else []
    latest = a.drop_duplicates("firm", keep="first")  # 증권사별 최신 의견
    bull_firms = latest[latest["stance"] == 1]["firm"].tolist()
    bear_firms = latest[latest["stance"] == -1]["firm"].tolist()
    ups, downs = int((a["action"] == "up").sum()), int((a["action"] == "down").sum())
    inits_bull = int(((a["action"] == "init") & (a["stance"] == 1)).sum())
    raises, lowers = int((a["ptdir"] == 1).sum()), int((a["ptdir"] == -1).sum())

    targets = targets or {}
    cur = targets.get("current")
    mean_t = targets.get("mean") or targets.get("median")
    upside = (mean_t / cur - 1) * 100 if cur and mean_t else np.nan

    sb = b = h = s = ss = 0
    if recs is not None and len(recs):
        r0 = recs.iloc[0]
        if "period" in recs.columns:
            m0 = recs[recs["period"] == "0m"]
            r0 = m0.iloc[0] if len(m0) else r0
        sb, b, h, s, ss = (int(r0.get(k, 0) or 0) for k in ("strongBuy", "buy", "hold", "sell", "strongSell"))
    tot = sb + b + h + s + ss
    buy_ratio = (sb + b) / tot * 100 if tot else np.nan
    sell_ratio = (s + ss) / tot * 100 if tot else np.nan

    mom = ups + raises + inits_bull - downs - lowers
    bull_score = (35 * min(len(bull_firms) / 8, 1)
                  + 25 * (np.nan_to_num(buy_ratio) / 100)
                  + 20 * float(np.clip(np.nan_to_num(upside) / 30, 0, 1))
                  + 20 * float(np.clip(mom / 6, 0, 1)))
    bear_score = (35 * min(len(bear_firms) / 4, 1)
                  + 25 * (np.nan_to_num(sell_ratio) / 30 if tot else 0)
                  + 20 * float(np.clip(-np.nan_to_num(upside) / 15, 0, 1))
                  + 20 * float(np.clip(-mom / 6, 0, 1)))
    return {
        "ticker": ticker, "bull_firms": len(bull_firms), "bear_firms": len(bear_firms),
        "firms": latest["firm"].nunique(), "bull_list": bull_firms, "bear_list": bear_firms,
        "ups": ups, "downs": downs, "raises": raises, "lowers": lowers, "inits": inits_bull,
        "buy_ratio": buy_ratio, "sell_ratio": sell_ratio, "n_analysts": tot,
        "current": cur, "mean_t": mean_t, "high_t": targets.get("high"), "low_t": targets.get("low"),
        "upside": upside, "bull_score": round(min(bull_score, 100), 1), "bear_score": round(min(bear_score, 100), 1),
        "actions": a,
    }


def action_comment(r, sym: str = "$") -> str:
    """증권사 액션 한 줄 코멘트 (자동 생성)."""
    act = KO_ACTION.get(str(r.get("action") or "").lower(), str(r.get("action") or ""))
    to, fr = ko_grade(r.get("to")), ko_grade(r.get("from"))
    grade = f"{fr} → {to}" if r.get("action") in ("up", "down") and r.get("from") else to
    pt = ""
    if pd.notna(r.get("pt")):
        if pd.notna(r.get("pt_prior")) and r["pt_prior"] != r["pt"]:
            arrow = "상향" if r["pt"] > r["pt_prior"] else "하향"
            pt = f", 목표가 {sym}{r['pt_prior']:,.0f}→{sym}{r['pt']:,.0f} {arrow}"
        else:
            pt = f", 목표가 {sym}{r['pt']:,.0f}"
    return f"{r['firm']}: {grade} ({act}){pt}"


def ticker_comment(sm: dict) -> str:
    """종목 단위 자동 코멘트."""
    parts = []
    if sm["bull_firms"]:
        parts.append(f"최근 매수 의견 증권사 {sm['bull_firms']}곳")
    if sm["bear_firms"]:
        parts.append(f"매도 의견 {sm['bear_firms']}곳")
    if sm["ups"] or sm["downs"]:
        parts.append(f"상향 {sm['ups']} / 하향 {sm['downs']}")
    if sm["raises"] or sm["lowers"]:
        parts.append(f"목표가 상향 {sm['raises']} / 하향 {sm['lowers']}")
    if pd.notna(sm["upside"]):
        parts.append(f"평균 목표가 기준 상승여력 {sm['upside']:+.0f}%")
    mood = ("월가 시각 **강한 낙관**" if sm["bull_score"] >= 70 else "월가 시각 **낙관**" if sm["bull_score"] >= 50
            else "월가 시각 **비관**" if sm["bear_score"] >= 50 else "월가 시각 **중립/혼재**")
    return f"{mood} — " + " · ".join(parts) if parts else mood


def analyst_ai_prompt(sm: dict, headlines: list[str]) -> str:
    acts = "\n".join(f"- {r['date']:%m/%d} {action_comment(r)}" for _, r in sm["actions"].head(15).iterrows())
    heads = "\n".join(f"- {h}" for h in headlines[:10]) or "- (없음)"
    return f"""너는 한국 개인투자자를 돕는 애널리스트다. {sm['ticker']}에 대한 최근 증권사 의견 데이터다.
컨센서스 매수비율 {sm['buy_ratio']:.0f}%, 평균 목표가 {sm['mean_t']}, 현재가 {sm['current']}, 상승여력 {sm['upside']:.0f}%.

[증권사 액션]
{acts}

[관련 뉴스 헤드라인]
{heads}

이 자료만 근거로 한국어 3문장 이내로: 월가가 이 종목을 보는 시각(강세/약세 논리), 목표가 흐름, 투자자가 주의할 점.
특정 애널리스트의 발언을 지어내지 말고, 데이터에 없는 내용은 쓰지 마라."""


ANALYST_NEWS_RE = re.compile(
    r"\b(analyst|upgrade|downgrade|price target|target|rating|overweight|underweight|outperform|underperform|"
    r"initiat|reiterat|coverage|buy rating|sell rating|PT)\b", re.I)


def is_analyst_headline(title: str) -> bool:
    return bool(ANALYST_NEWS_RE.search(title or ""))


UNIVERSE_CORE = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "LLY", "V", "MA",
                 "UNH", "XOM", "COST", "NFLX", "ORCL", "CRM", "AMD", "ADBE", "WMT", "BAC", "PLTR", "MU",
                 "UBER", "COIN", "PANW", "CRWD", "NOW", "ISRG"]
UNIVERSE_EXT = ["HD", "PG", "JNJ", "KO", "PEP", "MRK", "ABBV", "CVX", "TMO", "CSCO", "INTC", "QCOM", "TXN",
                "AMAT", "LRCX", "SHOP", "PYPL", "SNOW", "GE", "CAT", "BA", "DIS", "NKE", "SBUX", "GS", "MS",
                "ANET", "APP", "HOOD", "TSM"]

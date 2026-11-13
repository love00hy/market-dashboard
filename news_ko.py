"""
뉴스 한국어화
- translate_ko : 헤드라인 한국어 번역 (무료 Google 번역, 실패 시 None)
- tag_themes   : 헤드라인 → 테마(실적/금리/AI…) 분류
- rule_summary : API 키 없이 쓰는 규칙 기반 한눈 요약
- ai_summary   : Claude API 키가 있으면 한국어 요약·원인 분석
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

THEMES: list[tuple[str, list[str]]] = [
    ("📊 실적", ["earnings", "revenue", "guidance", "quarter", "profit", "eps", "results", "forecast", "outlook", "sales"]),
    ("🏦 금리·연준", ["fed", "rate", "powell", "inflation", "cpi", "treasury", "yield", "fomc", "pce", "bond"]),
    ("🤖 AI·반도체", ["ai", "artificial intelligence", "chip", "semiconductor", "gpu", "data center", "nvidia"]),
    ("🌐 관세·무역", ["tariff", "trade", "china", "export", "sanction", "import"]),
    ("⚖️ 규제·소송", ["sec", "lawsuit", "regulat", "probe", "antitrust", "court", "fine", "ban", "investigation"]),
    ("🤝 M&A·투자", ["acquire", "acquisition", "merger", "deal", "stake", "buyout", "takeover"]),
    ("🛢️ 원자재·유가", ["oil", "crude", "opec", "natural gas", "gold", "copper", "commodit"]),
    ("🎯 투자의견", ["upgrade", "downgrade", "price target", "analyst", "rating", "overweight", "underweight", "outperform"]),
    ("₿ 크립토", ["bitcoin", "crypto", "ether", "btc", "stablecoin", "blockchain", "token"]),
    ("🏛️ 정책·정치", ["trump", "congress", "election", "government", "shutdown", "white house", "senate", "bill"]),
    ("👥 경기·소비", ["jobs", "payroll", "unemployment", "consumer", "retail sales", "gdp", "recession", "spending", "housing"]),
]

_PATTERNS = [(label, re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")", re.I)) for label, kws in THEMES]


def tag_themes(text: str) -> list[str]:
    return [label for label, pat in _PATTERNS if pat.search(text or "")]


def translate_ko(texts: list[str]) -> list[str] | None:
    """무료 Google 번역(deep-translator). 실패하면 None."""
    if not texts:
        return []
    try:
        from deep_translator import GoogleTranslator
        tr = GoogleTranslator(source="auto", target="ko")
        out = tr.translate_batch(list(texts))
        return [o or t for o, t in zip(out, texts)]
    except Exception:
        return None


def rule_summary(news: pd.DataFrame) -> dict:
    """감성/테마 집계로 만드는 한눈 요약."""
    if news.empty:
        return {"tone": "😐 뉴스 없음", "themes": [], "bullets": []}
    s = news["sentiment"]
    pos, neg = int((s > 0).sum()), int((s < 0).sum())
    score = s.mean()
    tone = "😊 긍정 우세" if score > 0.15 else "😟 부정 우세" if score < -0.15 else "😐 긍정·부정 혼재"

    cnt: Counter = Counter()
    tone_by: dict[str, list[int]] = {}
    for _, r in news.iterrows():
        for th in tag_themes(f"{r['title']} {r.get('summary', '')}"):
            cnt[th] += 1
            tone_by.setdefault(th, []).append(int(r["sentiment"]))
    bullets = []
    for th, k in cnt.most_common(3):
        t = sum(tone_by[th])
        mood = "호재성" if t > 0 else "악재성" if t < 0 else "중립적"
        bullets.append(f"{th} 관련 뉴스 {k}건 — 대체로 {mood}")
    if not bullets:
        bullets.append("뚜렷한 공통 테마 없음 — 개별 종목 이슈 위주")
    return {"tone": f"{tone} (긍정 {pos} · 부정 {neg} · 전체 {len(news)}건)",
            "themes": [th for th, _ in cnt.most_common(5)], "bullets": bullets}


def ai_summary(api_key: str, prompt: str, model: str = "claude-haiku-5-5", max_tokens: int = 700) -> str | None:
    """Anthropic Messages API 호출. 실패하면 None."""
    if not api_key:
        return None
    try:
        import requests
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
    except Exception as e:  # 키 오류·한도 초과 등
        return f"⚠️ AI 요약 실패: {e}"


def _headline_block(news: pd.DataFrame, limit: int = 15) -> str:
    lines = []
    for _, r in news.head(limit).iterrows():
        summ = (r.get("summary") or "")[:200]
        lines.append(f"- [{r['ticker']}] {r['title']}" + (f" — {summ}" if summ else ""))
    return "\n".join(lines)


def sector_prompt(name: str, etf: str, period: str, chg: float, flow_text: str, news: pd.DataFrame) -> str:
    return f"""너는 한국 개인투자자를 돕는 미국 주식 애널리스트다.
미국 '{name}' 섹터(ETF {etf})는 최근 {period} 동안 {chg:+.2f}% 움직였고, 자금흐름 추정은 {flow_text}이다.
아래는 이 섹터 ETF와 대표 종목의 최신 뉴스 헤드라인이다.

{_headline_block(news)}

이 뉴스를 근거로 아래 형식 그대로, 한국어로 짧고 읽기 쉽게 써라. 뉴스에 근거가 없으면 지어내지 말고 "뚜렷한 뉴스 요인 없음"이라고 써라.

**한 줄 요약:** (이 섹터가 왜 {'강했는지' if chg >= 0 else '약했는지'} 한 문장)

**주요 원인**
- (원인 1, 관련 종목 티커 포함, 한 문장)
- (원인 2)
- (원인 3, 없으면 생략)

**체크 포인트:** (앞으로 지켜볼 변수 한 문장)"""


def ticker_prompt(ticker: str, chg_1m: float, news: pd.DataFrame, earnings: str) -> str:
    return f"""너는 한국 개인투자자를 돕는 미국 주식 애널리스트다.
{ticker} 종목의 최근 1개월 수익률은 {chg_1m:+.2f}%이고, 다음 실적발표 예정일은 {earnings}이다.
아래는 최신 뉴스 헤드라인이다.

{_headline_block(news, 20)}

이 뉴스를 근거로 아래 형식 그대로, 한국어로 짧고 읽기 쉽게 써라. 근거 없는 추측은 하지 마라.

**주요 이슈**
- (이슈 1 — 주가에 호재/악재 표시, 한 문장)
- (이슈 2)
- (이슈 3)
- (이슈 4, 없으면 생략)

**종합:** (지금 이 종목을 둘러싼 분위기 한 문장)"""


def market_prompt(rows: list[str], period: str) -> str:
    body = "\n".join(rows)
    return f"""너는 한국 개인투자자를 돕는 미국 주식 애널리스트다.
아래는 최근 {period} 미국 섹터 ETF 성과와 대표 뉴스 헤드라인이다.

{body}

한국어로 3~4문장으로 오늘 미국 시장의 섹터 로테이션(어디로 돈이 몰리고 어디서 빠지는지)과 그 이유를 요약하라. 근거 없는 추측은 하지 마라."""


def tone_score(news: pd.DataFrame) -> float:
    """뉴스 톤 점수 0~10 (5 = 중립). 최근 뉴스에 가중치."""
    if news.empty:
        return 5.0
    s = news["sentiment"].astype(float).values
    w = 0.85 ** pd.Series(range(len(s))).values  # 최신순 정렬 가정, 오래될수록 가중치↓
    mean = float((s * w).sum() / w.sum())
    return round(5 + 5 * mean, 1)


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

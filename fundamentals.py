"""
재무 기본지표 동종업계 비교 (Yahoo Finance info)
- 지표별로 경쟁사 사이 순위를 매겨 0~10점
- 분야(밸류에이션/수익성/성장성/재무건전성) 평균 → 종합 점수
- 강점·약점 자동 코멘트
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# key: (한글 이름, yfinance info 키, 분야, 높을수록 좋은가, 표시 형식)
METRICS: dict[str, tuple] = {
    "per":    ("PER (주가수익비율)", "trailingPE", "밸류에이션", False, "x"),
    "fper":   ("선행 PER", "forwardPE", "밸류에이션", False, "x"),
    "pbr":    ("PBR (주가순자산비율)", "priceToBook", "밸류에이션", False, "x"),
    "psr":    ("PSR (주가매출비율)", "priceToSalesTrailing12Months", "밸류에이션", False, "x"),
    "ev":     ("EV/EBITDA", "enterpriseToEbitda", "밸류에이션", False, "x"),
    "peg":    ("PEG", "trailingPegRatio", "밸류에이션", False, "x"),
    "roe":    ("ROE (자기자본이익률)", "returnOnEquity", "수익성", True, "%"),
    "roa":    ("ROA (총자산이익률)", "returnOnAssets", "수익성", True, "%"),
    "opm":    ("영업이익률", "operatingMargins", "수익성", True, "%"),
    "npm":    ("순이익률", "profitMargins", "수익성", True, "%"),
    "gpm":    ("매출총이익률", "grossMargins", "수익성", True, "%"),
    "rev_g":  ("매출 성장률 (YoY)", "revenueGrowth", "성장성", True, "%"),
    "eps_g":  ("이익 성장률 (YoY)", "earningsGrowth", "성장성", True, "%"),
    "de":     ("부채비율 (D/E)", "debtToEquity", "재무건전성", False, "%raw"),
    "cur":    ("유동비율", "currentRatio", "재무건전성", True, "x"),
    "fcf_y":  ("FCF 수익률", "_fcf_yield", "재무건전성", True, "%"),
}
CATEGORIES = ["밸류에이션", "수익성", "성장성", "재무건전성"]
# 밸류에이션 지표는 적자(음수)면 '가장 나쁨'으로 처리
NEG_IS_WORST = {"per", "fper", "ev", "peg", "pbr"}

FALLBACK_PEERS = {
    "COIN": ["HOOD", "SCHW", "IBKR", "MSTR", "SOFI", "PYPL", "XYZ"],
    "NVDA": ["AMD", "AVGO", "INTC", "QCOM", "MU", "TSM", "ARM"],
    "AAPL": ["MSFT", "GOOGL", "SONY", "DELL", "HPQ"],
    "TSLA": ["GM", "F", "RIVN", "TM", "LCID", "NIO"],
    "MSFT": ["ORCL", "CRM", "ADBE", "NOW", "GOOGL", "AMZN"],
}


def extract(info: dict) -> dict:
    """info → 지표 값 (FCF 수익률 계산 포함)."""
    out = {"name": info.get("shortName") or info.get("longName") or "",
           "sector": info.get("sector", ""), "industry": info.get("industry", ""),
           "mcap": info.get("marketCap"), "price": info.get("currentPrice") or info.get("regularMarketPrice"),
           "currency": info.get("currency", "USD")}
    for k, (_, key, *_rest) in METRICS.items():
        if key == "_fcf_yield":
            fcf, mc = info.get("freeCashflow"), info.get("marketCap")
            v = fcf / mc if fcf is not None and mc else None
        else:
            v = info.get(key)
        try:
            v = float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            v = np.nan
        out[k] = v
    return out


def score_peers(rows: dict[str, dict]) -> pd.DataFrame:
    """
    rows: {ticker: extract() 결과}. 지표별 0~10점 (동종 내 백분위).
    결측은 점수 없음(NaN), 밸류에이션 음수(적자)는 0점.
    """
    df = pd.DataFrame(rows).T
    for k, (_, _, _, higher, _) in METRICS.items():
        v = pd.to_numeric(df[k], errors="coerce")
        adj = v.copy()
        if k in NEG_IS_WORST:
            adj = adj.where(~(adj <= 0), np.inf)  # 적자/음수 → 최악
        valid = adj.notna()
        n = int(valid.sum())
        if n < 2:
            df[f"s_{k}"] = np.nan
            continue
        r = adj[valid].rank(ascending=not higher, method="average")  # 좋은 쪽이 rank 1
        s = (n - r) / (n - 1) * 10
        df[f"s_{k}"] = np.nan
        df.loc[valid, f"s_{k}"] = s
    for c in CATEGORIES:
        ks = [f"s_{k}" for k, m in METRICS.items() if m[2] == c]
        df[f"cat_{c}"] = df[ks].astype(float).mean(axis=1, skipna=True)
    df["total"] = df[[f"cat_{c}" for c in CATEGORIES]].astype(float).mean(axis=1, skipna=True)
    return df


def peer_median(df: pd.DataFrame, k: str) -> float:
    col = pd.to_numeric(df[k], errors="coerce")
    if k in NEG_IS_WORST:
        col = col[col > 0]
    return float(col.median()) if col.notna().any() else np.nan


def fmt_val(k: str, v) -> str:
    if v is None or pd.isna(v):
        return "-"
    f = METRICS[k][4]
    if f == "%":
        return f"{v * 100:.1f}%"
    if f == "%raw":
        return f"{v:.0f}%"
    return f"{v:.1f}배" if k not in ("cur",) else f"{v:.2f}"


def grade(s) -> str:
    if s is None or pd.isna(s):
        return "-"
    return "🟢 우수" if s >= 7 else "🟡 보통" if s >= 4 else "🔴 열위"


def metric_table(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    rows = []
    n_all = len(df)
    for k, (label, _, cat, higher, _) in METRICS.items():
        v = df.at[ticker, k]
        med = peer_median(df, k)
        s = df.at[ticker, f"s_{k}"]
        valid = df[f"s_{k}"].notna().sum()
        rank = int((df[f"s_{k}"].astype(float) > float(s)).sum() + 1) if pd.notna(s) else None
        rows.append({"분야": cat, "지표": label, "내 종목": fmt_val(k, v), "동종 중앙값": fmt_val(k, med),
                     "순위": f"{rank}/{valid}" if rank else "-", "점수": s, "평가": grade(s),
                     "기준": "높을수록 좋음" if higher else "낮을수록 좋음"})
    return pd.DataFrame(rows)


def comments(df: pd.DataFrame, ticker: str) -> list[str]:
    """강점·약점 자동 코멘트."""
    items = []
    for k, (label, _, cat, higher, _) in METRICS.items():
        s = df.at[ticker, f"s_{k}"]
        if pd.isna(s):
            continue
        med = peer_median(df, k)
        items.append((float(s), k, label, df.at[ticker, k], med, higher))
    if not items:
        return ["비교할 재무 데이터가 부족합니다."]
    items.sort(reverse=True)
    out = []
    tot = df.at[ticker, "total"]
    rank = int((df["total"].astype(float) > float(tot)).sum() + 1)
    cats = {c: df.at[ticker, f"cat_{c}"] for c in CATEGORIES}
    best = max((c for c in cats if pd.notna(cats[c])), key=lambda c: cats[c], default=None)
    worst = min((c for c in cats if pd.notna(cats[c])), key=lambda c: cats[c], default=None)
    out.append(f"동종 {len(df)}개 중 종합 **{rank}위** ({tot:.1f}/10) · 가장 강한 분야 **{best}**, 가장 약한 분야 **{worst}**")

    def why(k, v, med, higher):
        if k in NEG_IS_WORST and pd.notna(v) and v <= 0:
            return "적자라 산출 불가(최하점)"
        return f"{fmt_val(k, v)} vs 중앙값 {fmt_val(k, med)}"
    strengths = [f"{lb} ({why(k, v, md, h)})" for s, k, lb, v, md, h in items[:3] if s >= 6]
    weak = [f"{lb} ({why(k, v, md, h)})" for s, k, lb, v, md, h in items[::-1][:3] if s <= 4]
    if strengths:
        out.append("💪 강점: " + " · ".join(strengths))
    if weak:
        out.append("⚠️ 약점: " + " · ".join(weak))
    v = cats.get("밸류에이션")
    p = cats.get("수익성")
    if pd.notna(v) and pd.notna(p):
        if v >= 6 and p >= 6:
            out.append("➡️ 경쟁사보다 **싸면서 돈도 잘 버는** 편 (가치+퀄리티)")
        elif v < 4 and p >= 6:
            out.append("➡️ 수익성이 좋은 만큼 **프리미엄(비싼 가격)**을 받는 중")
        elif v >= 6 and p < 4:
            out.append("➡️ 싸 보이지만 수익성이 약함 — **가치 함정** 여부 확인 필요")
        elif v < 4 and p < 4:
            out.append("➡️ 비싸고 수익성도 낮음 — **성장 기대**가 주가에 반영된 상태")
    return out


def ai_prompt(df: pd.DataFrame, ticker: str) -> str:
    mt = metric_table(df, ticker)
    lines = "\n".join(f"- {r['지표']}: {r['내 종목']} (동종 중앙값 {r['동종 중앙값']}, 순위 {r['순위']})" for _, r in mt.iterrows())
    peers = ", ".join(df.index)
    return f"""너는 한국 개인투자자를 돕는 애널리스트다. {ticker}({df.at[ticker, 'name']}, 업종: {df.at[ticker, 'industry']})의
재무지표를 동종 경쟁사({peers})와 비교한 결과다. 종합 {df.at[ticker, 'total']:.1f}/10점.

{lines}

이 숫자만 근거로 한국어 4문장 이내로: 경쟁사 대비 이 회사의 위치(싼지/비싼지, 수익성·성장성·재무 안정성),
가장 눈에 띄는 강점과 약점, 투자 시 확인할 점. 데이터에 없는 내용은 쓰지 마라."""

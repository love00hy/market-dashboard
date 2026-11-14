"""
거장·기관 13F 포트폴리오 (SEC EDGAR, 무료 공식 데이터)
- 최신 2개 분기 13F-HR 보유내역 수집 → 분기 대비 증감
- CUSIP → 티커 변환 (OpenFIGI, 무료)
- 거장 간 중복 종목 점수
- 매매 내역 기반 자동 코멘트
"""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

import numpy as np
import pandas as pd

# 표시 이름 : (SEC CIK, 운용사)
GURUS: dict[str, tuple[int, str]] = {
    "워런 버핏": (1067983, "Berkshire Hathaway"),
    "스탠리 드러켄밀러": (1536411, "Duquesne Family Office"),
    "빌 애크먼": (1336528, "Pershing Square"),
    "데이비드 테퍼": (1656456, "Appaloosa"),
    "리루 (Li Lu)": (1709323, "Himalaya Capital"),
    "세스 클라만": (1061768, "Baupost Group"),
    "테리 스미스": (1569205, "Fundsmith"),
    "필립 라폰트": (1135730, "Coatue Management"),
    "대니얼 롭": (1040273, "Third Point"),
    "조지 소로스": (1029160, "Soros Fund Management"),
    "체이스 콜먼": (1167483, "Tiger Global"),
    "캐시 우드": (1697748, "ARK Invest"),
    "레이 달리오": (1350694, "Bridgewater (종목 1000개+, 느림)"),
}
DEFAULT_GURUS = ["워런 버핏", "스탠리 드러켄밀러", "빌 애크먼", "데이비드 테퍼",
                 "리루 (Li Lu)", "세스 클라만", "테리 스미스", "필립 라폰트"]

STATUS_ORDER = ["신규", "증가", "유지", "감소", "청산"]


# ─────────────────────────────────────────────
# SEC 수집
# ─────────────────────────────────────────────

def _get(session, url: str, ua: str, as_json: bool = False):
    for attempt in range(3):
        r = session.get(url, headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}, timeout=30)
        if r.status_code == 429:
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        time.sleep(0.12)  # SEC 초당 10회 제한 준수
        return r.json() if as_json else r.content
    r.raise_for_status()


def list_13f(session, cik: int, ua: str) -> tuple[str, list[dict]]:
    """(SEC 등록 기관명, [{accession, filed, period}] 최신순) — 원본 13F-HR만, 분기별 1개."""
    sub = _get(session, f"https://data.sec.gov/submissions/CIK{cik:010d}.json", ua, as_json=True)
    rec = sub.get("filings", {}).get("recent", {})
    out, seen = [], set()
    for form, acc, filed, period in zip(rec.get("form", []), rec.get("accessionNumber", []),
                                        rec.get("filingDate", []), rec.get("reportDate", [])):
        if form != "13F-HR" or period in seen:
            continue
        seen.add(period)
        out.append({"accession": acc, "filed": filed, "period": period})
    out.sort(key=lambda x: x["period"], reverse=True)
    return sub.get("name", ""), out


def parse_infotable(xml_bytes: bytes) -> pd.DataFrame:
    """13F information table XML → DataFrame (네임스페이스 무시)."""
    root = ET.fromstring(xml_bytes)
    rows = []
    for el in root.iter():
        if not el.tag.endswith("infoTable"):
            continue
        d = {}
        for child in el.iter():
            tag = child.tag.split("}")[-1]
            if child.text and child.text.strip():
                d[tag] = child.text.strip()
        rows.append({
            "name": d.get("nameOfIssuer", ""),
            "cls": d.get("titleOfClass", ""),
            "cusip": d.get("cusip", "").upper(),
            "value": float(d.get("value", 0) or 0),
            "shares": float(d.get("sshPrnamt", 0) or 0),
            "put_call": d.get("putCall", ""),
        })
    return pd.DataFrame(rows, columns=["name", "cls", "cusip", "value", "shares", "put_call"])


def fetch_holdings(session, cik: int, accession: str, ua: str) -> pd.DataFrame:
    acc = accession.replace("-", "")
    idx = _get(session, f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json", ua, as_json=True)
    items = idx.get("directory", {}).get("item", [])
    xmls = [it for it in items if it.get("name", "").lower().endswith(".xml")
            and it.get("name", "").lower() != "primary_doc.xml"]
    if not xmls:
        return parse_infotable(b"<x/>")

    def size(it):
        try:
            return int(it.get("size") or 0)
        except ValueError:
            return 0
    best = max(xmls, key=size)
    raw = _get(session, f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{quote(best['name'])}", ua)
    return parse_infotable(raw)


def aggregate(h: pd.DataFrame) -> pd.DataFrame:
    """옵션(풋/콜) 제외, CUSIP별 합산."""
    if h.empty:
        return pd.DataFrame(columns=["cusip", "name", "value", "shares"])
    stock = h[h["put_call"].fillna("") == ""]
    g = stock.groupby("cusip").agg(name=("name", "first"), value=("value", "sum"), shares=("shares", "sum"))
    return g.reset_index()


def compare(cur: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    """두 분기 비교 → 비중·주식수 변화·상태."""
    m = cur.merge(prev, on="cusip", how="outer", suffixes=("", "_prev"))
    m["name"] = m["name"].fillna(m["name_prev"])
    for c in ("value", "shares", "value_prev", "shares_prev"):
        m[c] = m[c].fillna(0.0)
    total = m["value"].sum()
    total_prev = m["value_prev"].sum()
    m["weight"] = m["value"] / total * 100 if total else 0
    m["weight_prev"] = m["value_prev"] / total_prev * 100 if total_prev else 0
    m["share_chg"] = np.where(m["shares_prev"] > 0, (m["shares"] / m["shares_prev"] - 1) * 100, np.nan)

    def status(r):
        if r["shares_prev"] == 0 and r["shares"] > 0:
            return "신규"
        if r["shares"] == 0 and r["shares_prev"] > 0:
            return "청산"
        if r["share_chg"] > 2:
            return "증가"
        if r["share_chg"] < -2:
            return "감소"
        return "유지"
    m["status"] = m.apply(status, axis=1)
    return m.drop(columns=["name_prev"]).sort_values("value", ascending=False).reset_index(drop=True)


def load_guru(session, cik: int, ua: str) -> dict:
    """최신·직전 분기 비교 결과."""
    entity, filings = list_13f(session, cik, ua)
    if not filings:
        return {"entity": entity, "error": "13F 공시 없음"}
    cur = aggregate(fetch_holdings(session, cik, filings[0]["accession"], ua))
    prev = aggregate(fetch_holdings(session, cik, filings[1]["accession"], ua)) if len(filings) > 1 \
        else cur.iloc[0:0]
    comp = compare(cur, prev)
    tv, pv = cur["value"].sum(), prev["value"].sum()
    return {
        "entity": entity, "period": filings[0]["period"], "filed": filings[0]["filed"],
        "prev_period": filings[1]["period"] if len(filings) > 1 else None,
        "total": float(tv), "total_prev": float(pv),
        "value_chg": float((tv / pv - 1) * 100) if pv else np.nan,
        "holdings": comp,
    }


# ─────────────────────────────────────────────
# CUSIP → 티커 (OpenFIGI)
# ─────────────────────────────────────────────

def map_cusips(session, cusips: list[str], cache: dict, api_key: str = "", budget_s: float = 40) -> dict:
    """cache(dict)를 채워서 반환. 실패한 건 None. 무료(키 없음): 요청당 10개, 분당 25회."""
    todo = [c for c in dict.fromkeys(cusips) if c and c not in cache]
    batch = 100 if api_key else 10
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key
    t0 = time.time()
    for i in range(0, len(todo), batch):
        if time.time() - t0 > budget_s:
            break
        chunk = todo[i:i + batch]
        jobs = [{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in chunk]
        try:
            r = session.post("https://api.openfigi.com/v3/mapping", json=jobs, headers=headers, timeout=20)
            if r.status_code == 429:
                time.sleep(6)
                r = session.post("https://api.openfigi.com/v3/mapping", json=jobs, headers=headers, timeout=20)
            r.raise_for_status()
            for c, res in zip(chunk, r.json()):
                data = res.get("data") or []
                eq = [d for d in data if (d.get("securityType2") or "").lower() in ("common stock", "depositary receipt")] or data
                tk = eq[0].get("ticker") if eq else None
                cache[c] = tk.replace("/", "-").replace(" ", "-") if tk else None
        except Exception:
            break
    return cache


# ─────────────────────────────────────────────
# 중복 종목 / 코멘트
# ─────────────────────────────────────────────

def overlap(results: dict[str, dict]) -> pd.DataFrame:
    """여러 거장이 보유한 종목 집계 + 점수."""
    rows = []
    for guru, res in results.items():
        h = res.get("holdings")
        if h is None or h.empty:
            continue
        for _, r in h[h["shares"] > 0].iterrows():
            rows.append({"cusip": r["cusip"], "name": r["name"], "guru": guru, "weight": r["weight"],
                         "status": r["status"], "value": r["value"]})
        for _, r in h[h["status"] == "청산"].iterrows():
            rows.append({"cusip": r["cusip"], "name": r["name"], "guru": guru, "weight": 0.0,
                         "status": "청산", "value": 0.0})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    held = df[df["status"] != "청산"]

    def agg(g):
        h = g[g["status"] != "청산"]
        return pd.Series({
            "name": g["name"].iloc[0],
            "holders": h["guru"].nunique(),
            "gurus": ", ".join(f"{x}({w:.1f}%)" for x, w in zip(h["guru"], h["weight"])),
            "avg_weight": h["weight"].mean() if len(h) else 0.0,
            "max_weight": h["weight"].max() if len(h) else 0.0,
            "buyers": int(g["status"].isin(["신규", "증가"]).sum()),
            "new": int((g["status"] == "신규").sum()),
            "sellers": int(g["status"].isin(["감소", "청산"]).sum()),
            "total_value": h["value"].sum(),
        })
    out = df.groupby("cusip").apply(agg, include_groups=False).reset_index()
    out = out[out["holders"] > 0] if not held.empty else out
    # 점수: 보유 거장 수가 핵심, 이번 분기 매수는 가점, 매도는 감점, 평균 비중 소폭 반영
    out["score"] = (out["holders"] * 10 + out["buyers"] * 5 + out["new"] * 3 - out["sellers"] * 4
                    + out["avg_weight"].clip(upper=20) * 0.5).round(1)
    return out.sort_values(["score", "holders"], ascending=False).reset_index(drop=True)


def _nm(r, tickers: dict) -> str:
    t = tickers.get(r["cusip"])
    return f"{t}" if t else str(r["name"]).title()


def guru_comment(res: dict, tickers: dict) -> list[str]:
    """이번 분기 매매로 만든 자동 코멘트 (실제 발언 아님)."""
    h = res.get("holdings")
    if h is None or h.empty:
        return ["보유내역을 불러오지 못했습니다."]
    lines = []
    vc = res.get("value_chg")
    n_new, n_up = int((h["status"] == "신규").sum()), int((h["status"] == "증가").sum())
    n_dn, n_out = int((h["status"] == "감소").sum()), int((h["status"] == "청산").sum())
    held = h[h["shares"] > 0]
    top5 = held.head(5)["weight"].sum()
    if pd.notna(vc):
        lines.append(f"포트폴리오 평가액 {vc:+.1f}% (전분기 대비) · 보유 {len(held)}종목, 상위 5종목 비중 {top5:.0f}%"
                     + (" → **집중 투자형**" if top5 >= 60 else " → 분산형" if top5 < 35 else ""))
    if n_new + n_up > n_dn + n_out + 2:
        tone = "**매수 우위** — 주식 비중을 늘리는 분기"
    elif n_dn + n_out > n_new + n_up + 2:
        tone = "**매도 우위** — 차익실현·현금화 성향"
    else:
        tone = "매수·매도 균형 — 종목 교체(리밸런싱) 위주"
    lines.append(f"{tone} (신규 {n_new} · 증가 {n_up} · 감소 {n_dn} · 청산 {n_out})")
    top = held.iloc[0] if len(held) else None
    if top is not None:
        lines.append(f"최대 보유: **{_nm(top, tickers)}** (비중 {top['weight']:.1f}%, {top['status']})")
    buys = h[h["status"].isin(["신규", "증가"])].copy()
    buys = buys.sort_values("weight", ascending=False).head(3)
    if len(buys):
        lines.append("주목할 매수: " + ", ".join(
            f"{_nm(r, tickers)}({'신규' if r['status'] == '신규' else f'+{r.share_chg:.0f}%'})" for _, r in buys.iterrows()))
    sells = h[h["status"].isin(["감소", "청산"])].sort_values("weight_prev", ascending=False).head(3)
    if len(sells):
        lines.append("주목할 매도: " + ", ".join(
            f"{_nm(r, tickers)}({'청산' if r['status'] == '청산' else f'{r.share_chg:.0f}%'})" for _, r in sells.iterrows()))
    return lines


def guru_ai_prompt(guru: str, res: dict, tickers: dict) -> str:
    h = res["holdings"]
    held = h[h["shares"] > 0].head(15)
    lines = [f"- {_nm(r, tickers)} {str(r['name']).title()}: 비중 {r['weight']:.1f}%, {r['status']}"
             + (f" ({r['share_chg']:+.0f}%)" if pd.notna(r['share_chg']) else "") for _, r in held.iterrows()]
    exits = h[h["status"] == "청산"].head(5)
    lines += [f"- {_nm(r, tickers)}: 전량 매도 (직전 비중 {r['weight_prev']:.1f}%)" for _, r in exits.iterrows()]
    return f"""너는 한국 개인투자자를 돕는 애널리스트다. 아래는 {guru}({res.get('entity')})의 13F 공시
({res.get('period')} 기준, 직전 분기 대비) 주요 보유·매매 내역이다.

{chr(10).join(lines)}

이 데이터만 근거로 한국어 3~4문장으로 해설하라: 이번 분기 매매의 큰 방향(어떤 섹터·테마를 늘리고 줄였는지),
눈에 띄는 결정 1~2개, 개인투자자가 참고할 점. {guru}가 실제로 한 말처럼 인용하지 말고, 공시 데이터 해석임을 분명히 하라."""

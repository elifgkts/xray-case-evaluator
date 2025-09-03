# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Streamlit – Xray Case Freshness (Store/Web Compare)

Amaç
- Jira/Xray CSV (;) içinden seçilen test caseleri **güncel ürün sayfalarıyla** karşılaştırarak
  ilgili özelliğin hâlâ var olup olmadığına dair bir sinyal üretir.
- Karşılaştırma kaynakları:
  1) Web site içerik(leri) (örn. https://fizy.com ...)
  2) Google Play (paket adı ile)
  3) Apple App Store (app id ile; iTunes lookup API)

Kurulum
  pip install streamlit pandas requests beautifulsoup4 google-play-scraper rapidfuzz

Çalıştırma
  streamlit run streamlit_app.py

Notlar
- Bu araç gerçek UI/functional test yapmaz; **public metadata** (açıklamalar, "Yenilikler", sayfa metinleri)
  üzerinden kelime ve yakın eşleşme (fuzzy) ile sinyal üretir.
- Son karar yine QA doğrulaması gerektirir.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process

try:
    # PyPI: google-play-scraper; module: google_play_scraper
    from google_play_scraper import app as gp_app
except Exception:
    gp_app = None  # graceful fallback

# ---------------------------
# Yardımcılar
# ---------------------------

STOPWORDS_TR = set("""
ve veya ile için gibi mı mi mu mü de da ki bu şu o bir birden daha çok çokça hemen artık yeni eski
hakkında üzerine kadar sonra önce şu an anlık tüm tümü genel sadece özel
""".split())

RED_FLAGS = ["deprecated", "kaldırıldı", "artık yok", "kaldırıl", "deaktif", "pasif", "legacy", "eski ekran"]


def find_col(cols: List[str], needle: str) -> str:
    low = needle.lower()
    for c in cols:
        if low in c.lower():
            return c
    return ""


def tokenize(s: str) -> List[str]:
    s = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ ]+", " ", s.lower(), flags=re.UNICODE)
    toks = [t for t in s.split() if t and t not in STOPWORDS_TR and len(t) > 2]
    return toks


def extract_keywords(summary: str, extra_terms: List[str] | None = None) -> List[str]:
    toks = tokenize(summary)
    if extra_terms:
        toks += [t.lower() for t in extra_terms if t]
    # benzersiz sırayı koru
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:20]  # aşırı genişlemeyi engelle

# ------------- Kaynaklar -------------

@dataclass
class SourceText:
    name: str
    where: str  # web/appstore/play
    text: str
    url: Optional[str] = None


def fetch_web_text(url: str, timeout: int = 15) -> SourceText:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    # görünür metinleri topla
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    host = urlparse(url).netloc
    return SourceText(name=f"Web:{host}", where="web", text=text[:40000], url=url)


def fetch_play(package: str, lang: str = "tr", country: str = "tr") -> Optional[SourceText]:
    if gp_app is None:
        return None
    try:
        data = gp_app(package, lang=lang, country=country)
        desc = (data.get("description") or "")
        changes = (data.get("recentChanges") or "")
        body = f"{desc}\n\nYenilikler:\n{changes}"
        return SourceText(name="Google Play", where="play", text=body[:40000], url=f"https://play.google.com/store/apps/details?id={package}")
    except Exception:
        return None


def fetch_appstore(app_id: str, country: str = "tr") -> Optional[SourceText]:
    # iTunes lookup API
    try:
        r = requests.get("https://itunes.apple.com/lookup", params={"id": app_id, "country": country}, timeout=15)
        r.raise_for_status()
        js = r.json()
        if js.get("resultCount", 0) > 0:
            it = js["results"][0]
            body = f"{it.get('description','')}\n\nYenilikler:\n{it.get('releaseNotes','')}"
            return SourceText(name="App Store", where="appstore", text=body[:40000], url=it.get("trackViewUrl"))
    except Exception:
        return None
    return None

# ------------- Eşleşme Skoru -------------

@dataclass
class MatchResult:
    present: bool
    score: int
    top_hits: List[Tuple[str, int]]  # (keyword, score)
    redflags: List[str]
    evidence: Optional[str]
    source: str
    url: Optional[str]


def score_against_source(keywords: List[str], source: SourceText, threshold: int = 70) -> MatchResult:
    # fuzzy: her anahtar için token_set_ratio ile en iyi eşleşmeyi ölç
    hits: List[Tuple[str, int]] = []
    for kw in keywords:
        sc = fuzz.token_set_ratio(kw, source.text)
        if sc >= 40:  # düşük eşikler de raporlansın
            hits.append((kw, int(sc)))
    hits.sort(key=lambda x: x[1], reverse=True)
    present = any(sc >= threshold for _, sc in hits)

    # red flags
    reds = [rf for rf in RED_FLAGS if rf in source.text.lower()]

    # küçük bir kanıt snippet'i
    snippet = None
    if hits:
        best_kw, _ = hits[0]
        # ilk eşleşen segmenti çek
        m = re.search(re.escape(best_kw), source.text, re.IGNORECASE)
        if m:
            start = max(0, m.start() - 120)
            end = min(len(source.text), m.end() + 180)
            snippet = source.text[start:end]

    return MatchResult(present=present, score=hits[0][1] if hits else 0, top_hits=hits[:5], redflags=reds, evidence=snippet, source=source.name, url=source.url)


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Xray Case Freshness – Store/Web Compare", page_icon="🔎", layout="wide")
st.title("🔎 Xray Case Freshness – Store/Web Compare")
st.caption("CSV → Seçili caseleri Web / Play Store / App Store açıklamalarıyla karşılaştırarak güncellik sinyali üretir.")

uploaded = st.file_uploader("CSV yükle (Jira/Xray export; ; ile ayrılmış)", type=["csv"]) 

with st.expander("Kaynak ayarları"):
    web_urls = st.text_area("Web URL listesi (satır başına bir adres)", value="https://fizy.com\nhttps://fizy.com/kampanyalar")
    pkg_android = st.text_input("Google Play paket adı (ör. com.turkcell.bip gibi)", value="")
    appstore_id = st.text_input("App Store App ID (sayısal)", value="")
    lang_country = st.selectbox("Dil/Ülke (Play & App Store)", ["tr/TR", "en/US"], index=0)
    thr = st.slider("Eşleşme eşiği (fuzzy)", min_value=60, max_value=90, value=70, step=1)

if uploaded is None:
    st.info("CSV yükleyin. Gerekli sütunlar: Issue key/Key, Summary. Opsiyonel: Manual Test Steps, Project/Labels/Components.")
    st.stop()

# CSV oku
try:
    df_raw = pd.read_csv(uploaded, sep=";", dtype=str, low_memory=False)
except Exception:
    df_raw = pd.read_csv(uploaded, dtype=str, low_memory=False)

col_key = find_col(df_raw.columns.tolist(), "Issue key") or find_col(df_raw.columns.tolist(), "Key")
col_sum = find_col(df_raw.columns.tolist(), "Summary")
if not col_key or not col_sum:
    st.error("Issue key/Key ve Summary sütunları gerekli.")
    st.stop()

# Örneklem kontrolleri
c1, c2 = st.columns(2)
with c1:
    sample_n = st.number_input("Örneklem (adet)", min_value=1, max_value=max(1, len(df_raw)), value=min(10, len(df_raw)), step=1)
with c2:
    seed = st.number_input("Rastgele seed", min_value=0, max_value=99999, value=42, step=1)

if len(df_raw) > sample_n:
    df = df_raw.sample(n=int(sample_n), random_state=int(seed))
else:
    df = df_raw.copy()

# Kaynakları getir
lang, country = lang_country.split("/")

sources: List[SourceText] = []
# Web
urls = [u.strip() for u in web_urls.splitlines() if u.strip()]
for url in urls:
    try:
        sources.append(fetch_web_text(url))
    except Exception as e:
        st.warning(f"Web kaynağı alınamadı: {url} → {e}")

# Play
if pkg_android:
    st.caption(":information_source: Google Play içeriği (paket adı gerek): description + recentChanges")
    src = fetch_play(pkg_android, lang=lang, country=country)
    if src:
        sources.append(src)
    else:
        st.warning("Google Play verisi alınamadı veya paket geçersiz.")

# App Store
if appstore_id:
    st.caption(":information_source: App Store içeriği (App ID gerek): description + releaseNotes")
    src = fetch_appstore(appstore_id, country=country)
    if src:
        sources.append(src)
    else:
        st.warning("App Store verisi alınamadı veya App ID geçersiz.")

if not sources:
    st.error("Hiç kaynak alınamadı. En az bir web adresi veya mağaza kimliği giriniz.")
    st.stop()

# Eşle ve karar ver
rows: List[Dict[str, Any]] = []
for _, r in df.iterrows():
    key = r.get(col_key)
    summary = r.get(col_sum) or ""
    kws = extract_keywords(summary)
    match_details: List[Dict[str, Any]] = []
    any_present = False
    any_red = False

    for src in sources:
        mr = score_against_source(kws, src, threshold=int(thr))
        any_present = any_present or mr.present
        any_red = any_red or bool(mr.redflags)
        match_details.append({
            "Source": mr.source,
            "URL": mr.url or "",
            "Present": "Evet" if mr.present else "Belirsiz",
            "Score": mr.score,
            "Top hits": ", ".join([f"{w}:{s}" for w,s in mr.top_hits]),
            "Red flags": ", ".join(mr.redflags),
            "Evidence": (mr.evidence or "")[:300]
        })

    # Karar mantığı
    if any_present and not any_red:
        evaluation = "Evet"
    elif any_red and not any_present:
        evaluation = "Hayır"
    else:
        evaluation = "Evet*"  # sinyaller karışık veya yetersiz

    rows.append({
        "Issue key": key,
        "Summary": summary,
        "Evaluation": evaluation,
        "Details": json.dumps(match_details, ensure_ascii=False)
    })

out = pd.DataFrame(rows)

st.subheader("Sonuçlar")
st.dataframe(out[["Issue key","Summary","Evaluation"]], use_container_width=True)

with st.expander("Kaynak eşleşme detayları"):
    st.dataframe(out[["Issue key","Details"]], use_container_width=True)

# İndirme

def df_to_csv_bom(df: pd.DataFrame, sep: str = ";") -> bytes:
    s = df.to_csv(index=False, sep=sep, encoding="utf-8-sig")
    return s.encode("utf-8-sig")

st.download_button(
    label="CSV indir (UTF-8 BOM, ;)",
    data=df_to_csv_bom(out, sep=";"),
    file_name="store_web_compare_results.csv",
    mime="text/csv",
)

st.caption("Not: Bu yöntem store/web açıklamalarına dayanır; tüm özellikleri listelemeyebilir. Son karar için uygulama içi test önerilir.")

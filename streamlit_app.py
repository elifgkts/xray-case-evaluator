# -*- coding: utf-8 -*-
"""
Xray Case Freshness – Environment-Aware (Store/Web Compare)

Yenilikler:
- Ürün seçici eklendi: **Fizy** ve **Game+** (manuel mod da mevcut).
- Issue key ipucu: QF* → Web, QB* → Backend (öncelikli kural)
- Google Play paket adı ve App Store App ID alanları ürün seçimine göre otomatik dolar:
    • Fizy varsayılanları: GP=com.turkcell.gncplay, iOS=404239912
    • Game+: Mobil uygulama yok → Store seçenekleri "Yok (uygulanmaz)"
- Web URL listesi ürün seçimine göre otomatik dolar (Fizy/Game+ presetleri). Manuel düzenlenebilir.

Not:
- Bu bir fonksiyonel test değildir; public metinlerden sinyal üretir. QA doğrulaması tavsiye edilir.
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

# ---- Opsiyonel bağımlılıklar (fallback'lı) ----
try:
    from bs4 import BeautifulSoup
    HAVE_BS4 = True
except Exception:
    HAVE_BS4 = False

try:
    from rapidfuzz import fuzz as _rf_fuzz
    def FUZZ(a: str, b: str) -> int:
        return int(_rf_fuzz.token_set_ratio(a, b))
except Exception:
    # rapidfuzz yoksa difflib ile basit skor
    from difflib import SequenceMatcher
    def FUZZ(a: str, b: str) -> int:
        return int(100 * SequenceMatcher(None, a, b).ratio())

try:
    from google_play_scraper import app as gp_app
except Exception:
    gp_app = None  # yoksa Play kaynağını atlarız

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
    s = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ ]+", " ", (s or "").lower(), flags=re.UNICODE)
    toks = [t for t in s.split() if t and t not in STOPWORDS_TR and len(t) > 2]
    return toks

def extract_keywords(summary: str, extra_terms: List[str] | None = None) -> List[str]:
    toks = tokenize(summary or "")
    if extra_terms:
        toks += [t.lower() for t in extra_terms if t]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:20]

def html_to_text(html: str) -> str:
    if HAVE_BS4:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script","style","noscript"]): tag.decompose()
        return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    # fallback: kaba temizleme
    txt = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    txt = re.sub(r"<style[\s\S]*?</style>", " ", txt, flags=re.I)
    txt = re.sub(r"<[^>]+>", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

# ------------- Kaynak Okuyucular -------------

@dataclass
class SourceText:
    name: str
    where: str   # web/play/appstore
    text: str
    url: Optional[str] = None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_web_text(url: str, timeout: int = 15) -> SourceText:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    text = html_to_text(r.text)
    host = urlparse(url).netloc
    return SourceText(name=f"Web:{host}", where="web", text=text[:40000], url=url)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_play(package: str, lang: str = "tr", country: str = "tr") -> Optional[SourceText]:
    if gp_app is None or not package:
        return None
    try:
        data = gp_app(package, lang=lang, country=country)
        desc = (data.get("description") or "")
        changes = (data.get("recentChanges") or "")
        body = f"{desc}\n\nYenilikler:\n{changes}"
        return SourceText(name="Google Play", where="play", text=body[:40000],
                          url=f"https://play.google.com/store/apps/details?id={package}")
    except Exception:
        return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_appstore(app_id: str, country: str = "tr") -> Optional[SourceText]:
    if not app_id:
        return None
    try:
        r = requests.get("https://itunes.apple.com/lookup",
                         params={"id": app_id, "country": country}, timeout=15)
        r.raise_for_status()
        js = r.json()
        if js.get("resultCount", 0) > 0:
            it = js["results"][0]
            body = f"{it.get('description','')}\n\nYenilikler:\n{it.get('releaseNotes','')}"
            return SourceText(name="App Store", where="appstore", text=body[:40000],
                              url=it.get("trackViewUrl"))
    except Exception:
        return None
    return None

# ------------- Eşleştirme -------------

@dataclass
class MatchResult:
    present: bool
    score: int
    top_hits: List[Tuple[str, int]]
    redflags: List[str]
    evidence: Optional[str]
    source: str
    url: Optional[str]

def score_against_source(keywords: List[str], source: SourceText, threshold: int = 70) -> MatchResult:
    hits: List[Tuple[str, int]] = []
    for kw in keywords:
        sc = FUZZ(kw, source.text)
        if sc >= 40:  # düşük eşikler de raporlansın
            hits.append((kw, int(sc)))
    hits.sort(key=lambda x: x[1], reverse=True)
    present = any(sc >= threshold for _, sc in hits)
    reds = [rf for rf in RED_FLAGS if rf in source.text.lower()]

    snippet = None
    if hits:
        best_kw, _ = hits[0]
        m = re.search(re.escape(best_kw), source.text, re.IGNORECASE)
        if m:
            start = max(0, m.start() - 120)
            end = min(len(source.text), m.end() + 180)
            snippet = source.text[start:end]

    return MatchResult(
        present=present,
        score=hits[0][1] if hits else 0,
        top_hits=hits[:5],
        redflags=reds,
        evidence=snippet,
        source=source.name,
        url=source.url
    )

# ------------- Ortam çıkarımı -------------

def infer_target_env(platform_val: str | None, components_val: str | None, issue_key: str | None = "") -> str:
    # 1) Issue key ipuçları (öncelikli)
    if issue_key:
        s = str(issue_key)
        if s.startswith("QF"):
            return "Web"
        if s.startswith("QB"):
            return "Backend"

    # 2) Platform/Component metinleri
    p = (platform_val or "").lower()
    c = (components_val or "").lower()
    text = f"{p} {c}"
    if any(k in text for k in ["ios", "iphone", "ipad"]):
        return "iOS"
    if any(k in text for k in ["android", "apk"]):
        return "Android"
    if any(k in text for k in ["mobile", "mobil"]):
        return "Mobile"
    if any(k in text for k in ["web", "frontend", "fe", "ui-web"]):
        return "Web"
    if any(k in text for k in ["backend", "server", "api", "db", "oracle"]):
        return "Backend"
    return "Unknown"

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Xray Case Freshness – Env Aware", page_icon="🔎", layout="wide")
st.title("🔎 Xray Case Freshness – Environment-Aware Compare")
st.caption("CSV → Hedef ortama göre (Web/Play/App Store) karşılaştırma + fuzzy skor + kanıt")

uploaded = st.file_uploader("CSV yükle (Jira/Xray export; ; ile ayrılmış)", type=["csv"])

# --- Ürün seçici & presetler ---
with st.expander("Ürün seçimi ve presetler"):
    PRODUCT_OPTIONS = ["Fizy", "Game+", "Özel (manuel)"]
    product = st.selectbox("Ürün", options=PRODUCT_OPTIONS, index=0, help="Seçime göre web/store alanları otomatik dolar.")

    # Preset web URL listeleri
    PRESETS_WEB: Dict[str, str] = {
        "Fizy": "https://fizy.com\nhttps://fizy.com/kampanyalar",
        "Game+": "https://gameplus.com.tr\nhttps://gameplus.com.tr/blog\nhttps://gameplus.com.tr/firsatlar\nhttps://gameplus.com.tr/destek",
        "Özel (manuel)": "",
    }

    # Preset store kimlikleri
    PRESETS_STORE: Dict[str, Dict[str, List[str]]] = {
        "Fizy": {
            "play": ["com.turkcell.gncplay", "Manuel giriş", "Yok (uygulanmaz)"],
            "ios": ["404239912", "Manuel giriş", "Yok (uygulanmaz)"],
        },
        "Game+": {
            # Game+'ta mobil uygulama yok
            "play": ["Yok (uygulanmaz)", "Manuel giriş"],

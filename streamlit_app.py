# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Streamlit – Xray Case Freshness (Environment‑Aware Store/Web Compare)

Amaç
- Jira/Xray CSV (;) içindeki caseleri **hedef ortamına göre** doğru kanalla
  (Web / Google Play / App Store) karşılaştırıp güncellik sinyali üretmek.
- Hedef ortamı CSV'deki **Component/s** ve **Custom field (Platform)** alanlarından çıkarır;
  kullanıcı isterse satır bazında override edebilir.

Kurulum (requirements.txt)
  streamlit
  pandas
  requests
  beautifulsoup4
  google-play-scraper
  rapidfuzz

Çalıştırma
  streamlit run streamlit_app.py

Not
- Bu araç public metinler (web sayfası, mağaza açıklamaları) üzerinden fuzzy eşleşme yapar;
  gerçek fonksiyonel test yerine **sinyal** üretir. QA doğrulaması tavsiye edilir.
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
from rapidfuzz import fuzz

try:
    from google_play_scraper import app as gp_app
except Exception:  # modül yüklü değilse uygulama yine çalışsın
    gp_app = None

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
    toks = tokenize(summary)
    if extra_terms:
        toks += [t.lower() for t in extra_terms if t]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t); out.append(t)
    return out[:20]

# ------------- Kaynak Okuyucular -------------

@dataclass
class SourceText:
    name: str
    where: str  # web/play/appstore
    text: str
    url: Optional[str] = None


def fetch_web_text(url: str, timeout: int = 15) -> SourceText:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    host = urlparse(url).netloc
    return SourceText(name=f"Web:{host}", where="web", text=text[:40000], url=url)


def fetch_play(package: str, lang: str = "tr", country: str = "tr") -> Optional[SourceText]:
    if gp_app is None or not package:
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
    if not app_id:
        return None
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
        sc = fuzz.token_set_ratio(kw, source.text)
        if sc >= 40:
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

    return MatchResult(present=present, score=hits[0][1] if hits else 0, top_hits=hits[:5], redflags=reds, evidence=snippet, source=source.name, url=source.url)

# ------------- Ortam çıkarımı -------------

def infer_target_env(platform_val: str | None, components_val: str | None) -> str:
    p = (platform_val or "").lower()
    c = (components_val or "").lower()
    text = f"{p} {c}"
    # iOS vs Android önceliği
    if any(k in text for k in ["ios", "iphone", "ipad"]):
        return "iOS"
    if any(k in text for k in ["android", "apk"]):
        return "Android"
    # genel mobile
    if any(k in text for k in ["mobile", "mobil"]):
        return "Mobile"
    # web
    if any(k in text for k in ["web", "frontend", "fe", "ui-web"]):
        return "Web"
    # backend
    if any(k in text for k in ["backend", "server", "api", "db", "oracle"]):
        return "Backend"
    return "Unknown"

# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Xray Case Freshness – Env Aware", page_icon="🔎", layout="wide")
st.title("🔎 Xray Case Freshness – Environment‑Aware Compare")
st.caption("CSV → Hedef ortama göre (Web/Play/App Store) karşılaştırma + fuzzy skor + detaylı kanıt")

uploaded = st.file_uploader("CSV yükle (Jira/Xray export; ; ile ayrılmış)", type=["csv"]) 

with st.expander("Kaynak & Parametre Ayarları"):
    web_urls = st.text_area("Web URL listesi (satır başına bir adres)", value="https://fizy.com\nhttps://fizy.com/kampanyalar")
    pkg_android_default = st.text_input("Google Play paket adı (örn. com.turkcell.gncplay)", value="")
    appstore_id_default = st.text_input("App Store App ID (sayısal)", value="")
    lang_country = st.selectbox("Dil/Ülke (Play & App Store)", ["tr/TR", "en/US"], index=0)
    thr = st.slider("Eşleşme eşiği (fuzzy)", min_value=60, max_value=90, value=70, step=1)

if uploaded is None:
    st.info("CSV yükleyin. Gerekli sütunlar: Issue key/Key, Summary. Ortam çıkarımı için: Component/s ve/veya Custom field (Platform)")
    st.stop()

# CSV oku
try:
    df_raw = pd.read_csv(uploaded, sep=";", dtype=str, low_memory=False)
except Exception:
    df_raw = pd.read_csv(uploaded, dtype=str, low_memory=False)

# Sütun tespiti
col_key = find_col(df_raw.columns.tolist(), "Issue key") or find_col(df_raw.columns.tolist(), "Key")
col_sum = find_col(df_raw.columns.tolist(), "Summary")
col_comp = find_col(df_raw.columns.tolist(), "Component/s") or find_col(df_raw.columns.tolist(), "Components")
col_platform = find_col(df_raw.columns.tolist(), "Custom field (Platform)") or find_col(df_raw.columns.tolist(), "Platform")

if not col_key or not col_sum:
    st.error("Issue key/Key ve Summary sütunları zorunlu.")
    st.stop()

# Örneklem
c1, c2 = st.columns(2)
with c1:
    sample_n = st.number_input("Örneklem (adet)", min_value=1, max_value=max(1, len(df_raw)), value=min(10, len(df_raw)), step=1)
with c2:
    seed = st.number_input("Rastgele seed", min_value=0, max_value=99999, value=42, step=1)

if len(df_raw) > sample_n:
    df = df_raw.sample(n=int(sample_n), random_state=int(seed))
else:
    df = df_raw.copy()

# Hedef ortam çıkarımı
env_series = df.apply(lambda r: infer_target_env(r.get(col_platform), r.get(col_comp)), axis=1)
work = df[[col_key, col_sum, col_comp, col_platform]].copy()
work.insert(2, "Target Env", env_series)
work = work.rename(columns={col_key: "Issue key", col_sum: "Summary", col_comp: "Component/s", col_platform: "Platform"})

st.subheader("Örneklem ve Hedef Ortam (düzenlenebilir)")
st.caption("Satır bazında hedef ortamı değiştirebilirsiniz. Android/iOS için paket/ID boşsa, genel ayarlardaki değerler kullanılır.")

# Kullanıcı override tablosu
edited = st.data_editor(
    work,
    column_config={
        "Target Env": st.column_config.SelectboxColumn(options=["Android","iOS","Mobile","Web","Backend","Unknown"], help="Bu case hangi ortamda doğrulanmalı?"),
    },
    use_container_width=True,
)

# Kaynakları hazırla (genel)
lang, country = lang_country.split("/")
web_list = [u.strip() for u in web_urls.splitlines() if u.strip()]

# Eşleştirme – satır bazında sadece ilgili kaynaklarda
rows: List[Dict[str, Any]] = []
for _, r in edited.iterrows():
    key = r.get("Issue key")
    summary = r.get("Summary") or ""
    target = (r.get("Target Env") or "Unknown").strip()

    # Kaynaklar bu satır için
    sources: List[SourceText] = []
    if target in ("Web", "Backend", "Unknown", "Mobile"):  # Mobile genel ise web de sinyal olabilir
        for url in web_list:
            try:
                sources.append(fetch_web_text(url))
            except Exception:
                pass
    if target in ("Android", "Mobile") and pkg_android_default:
        src = fetch_play(pkg_android_default, lang=lang, country=country)
        if src: sources.append(src)
    if target in ("iOS", "Mobile") and appstore_id_default:
        src = fetch_appstore(appstore_id_default, country=country)
        if src: sources.append(src)

    if not sources:
        rows.append({
            "Issue key": key,
            "Summary": summary,
            "Target Env": target,
            "Evaluation": "Evet*",
            "Reason": "Kaynak yok (URL/paket/ID giriniz)",
            "Details": json.dumps([], ensure_ascii=False)
        })
        continue

    # Anahtar kelimeler
    kws = extract_keywords(summary)

    # Kaynaklara karşı skorla
    any_present, any_red = False, False
    match_details: List[Dict[str, Any]] = []
    best_src = None
    best_score = -1

    for src in sources:
        mr = score_against_source(kws, src, threshold=int(thr))
        any_present = any_present or mr.present
        any_red = any_red or bool(mr.redflags)
        if mr.score > best_score:
            best_score = mr.score; best_src = mr
        match_details.append({
            "Source": mr.source,
            "URL": mr.url or "",
            "Present": "Evet" if mr.present else "Belirsiz",
            "Score": mr.score,
            "Top hits": ", ".join([f"{w}:{s}" for w,s in mr.top_hits]),
            "Red flags": ", ".join(mr.redflags),
            "Evidence": (mr.evidence or "")[:300]
        })

    # Karar
    if any_present and not any_red:
        evaluation = "Evet"
        reason = f"Kaynakta güçlü eşleşme (best {best_score})"
    elif any_red and not any_present:
        evaluation = "Hayır"
        reason = "Red-flag bulundu; güçlü eşleşme yok"
    else:
        evaluation = "Evet*"
        reason = f"Karışık/zayıf sinyal (best {best_score})"

    rows.append({
        "Issue key": key,
        "Summary": summary,
        "Target Env": target,
        "Evaluation": evaluation,
        "Reason": reason,
        "Details": json.dumps(match_details, ensure_ascii=False)
    })

out = pd.DataFrame(rows)

# Görünüm ve indirme
st.subheader("Sonuçlar")
st.dataframe(out[["Issue key","Target Env","Evaluation","Reason"]], use_container_width=True)

with st.expander("Kaynak eşleşme detayları"):
    st.dataframe(out[["Issue key","Details"]], use_container_width=True)


def df_to_csv_bom(df: pd.DataFrame, sep: str = ";") -> bytes:
    s = df.to_csv(index=False, sep=sep, encoding="utf-8-sig")
    return s.encode("utf-8-sig")

st.download_button(
    label="CSV indir (UTF-8 BOM, ;)",
    data=df_to_csv_bom(out, sep=";"),
    file_name="env_aware_compare_results.csv",
    mime="text/csv",
)

st.caption("Not: Hedef ortam satır bazında düzenlenebilir. Android/iOS için paket/ID girilmezse o kanallar devreye girmez.")

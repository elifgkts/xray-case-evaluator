# streamlit_app.py
# -*- coding: utf-8 -*-
"""
Streamlit – Xray Case Evaluator 

Amaç
- Jira/Xray CSV (;) içinden rastgele case seçip güncellik durumunu (Evet/Hayır/Evet*) heuristik olarak değerlendirmek
- Her case için "Environment" (Mobile/Web/Backend/Unknown) tespiti + kullanıcı tarafından düzeltilebilir olması
- Bu koşuda kontrolün hangi ortamda yapıldığını ayrıca kayıt etmek ("Checked On")

Çalıştırma
  streamlit run streamlit_app.py

Gereksinimler
  pip install streamlit pandas

Notlar
- Bu araç **step parser** değildir; ayrı bir uygulama olarak tasarlanmıştır.
- Heuristik kararlar QA gözden geçirmesi ile doğrulanmalıdır.
"""

from datetime import datetime
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ---------------------------
# Yardımcılar
# ---------------------------

def find_col(cols: List[str], needle: str) -> str:
    low = needle.lower()
    for c in cols:
        if low in c.lower():
            return c
    return ""


def parse_date_any(s: Optional[str]) -> Optional[datetime]:
    if not isinstance(s, str) or not s.strip():
        return None
    fmts = ["%d-%b-%y", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"]
    for f in fmts:
        try:
            return datetime.strptime(s[:len(f)], f)
        except Exception:
            continue
    try:
        dt = pd.to_datetime(s, errors="coerce")
        return None if pd.isna(dt) else dt.to_pydatetime()
    except Exception:
        return None


def months_diff(d1: datetime, d2: datetime) -> int:
    return (d1.year - d2.year) * 12 + (d1.month - d2.month)


# Basit ortam çıkarımı (kullanıcı sonradan düzenleyebilir)

def infer_environment(project: Optional[str], labels: Optional[str], components: Optional[str]) -> str:
    p = (project or "").lower()
    l = (labels or "").lower()
    c = (components or "").lower()
    text = f"{p} {l} {c}"
    if any(k in text for k in ["mobil", "mobile", "android", "ios"]):
        return "Mobile"
    if any(k in text for k in ["web", "frontend", "fe", "ui-web"]):
        return "Web"
    if any(k in text for k in ["backend", "server", "api", "db", "oracle"]):
        return "Backend"
    return "Unknown"


# Heuristik karar kuralları
RED_FLAGS = [
    "deprecated", "kaldırıldı", "legacy", "artık yok", "deaktif", "pasif", "eski ekran", "v2 kapandı"
]


def has_any_expected(steps_cell: Optional[str]) -> bool:
    if not isinstance(steps_cell, str) or not steps_cell.strip():
        return False
    # Basit kontrol: JSON parse etmeyi dener; olmazsa string içinde anahtar arar
    try:
        arr = json.loads(steps_cell.strip())
        if isinstance(arr, list):
            for it in arr:
                fields = it.get("fields", {}) if isinstance(it, dict) else {}
                if (fields.get("Expected Result") or "").strip():
                    return True
    except Exception:
        return "Expected Result" in steps_cell
    return False


def evaluate_case(summary: str, steps: Optional[str], created: Optional[str]) -> Tuple[str, str]:
    text_low = (summary or "").lower()
    created_dt = parse_date_any(created)
    now = datetime.now()
    score = 0
    reasons: List[str] = []

    # Tarih
    if created_dt:
        m = months_diff(now, created_dt)
        if m <= 18:
            score += 2; reasons.append(f"Tarih yeni (~{m} ay)")
        elif m >= 36:
            score -= 2; reasons.append(f"Tarih eski (~{m} ay)")

    # Red-flag
    if any(flag in text_low for flag in RED_FLAGS):
        score -= 3; reasons.append("Metinde deprecated/kaldırıldı vb.")

    # Expected Result
    if has_any_expected(steps):
        score += 1; reasons.append("Expected Result var")

    decision = "Evet" if score > 0 else ("Hayır" if score < 0 else "Evet*")
    return decision, "; ".join(reasons)


def df_to_csv_bom(df: pd.DataFrame, sep: str = ";") -> bytes:
    s = df.to_csv(index=False, sep=sep, encoding="utf-8-sig")
    return s.encode("utf-8-sig")


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="Xray Case Evaluator", page_icon="✅", layout="wide")
st.title("Xray Case Evaluator")
st.caption("CSV → Rastgele senaryo seçimi + Güncellik değerlendirmesi; ortam belirleme ve raporlama")

uploaded = st.file_uploader("CSV yükle (Jira/Xray export, ; ile ayrılmış)", type=["csv"]) 

if uploaded is None:
    st.info("CSV yükleyin. Gerekli sütunlar: Issue key/Key, Summary, Created, Project/Project name; opsiyonel: Labels, Components, Manual Test Steps")
    st.stop()

# CSV oku
try:
    df_raw = pd.read_csv(uploaded, sep=";", dtype=str, low_memory=False)
except Exception:
    df_raw = pd.read_csv(uploaded, dtype=str, low_memory=False)

# Sütun tespiti
col_key   = find_col(df_raw.columns.tolist(), "Issue key") or find_col(df_raw.columns.tolist(), "Key")
col_sum   = find_col(df_raw.columns.tolist(), "Summary")
col_steps = find_col(df_raw.columns.tolist(), "Manual Test Steps")
col_created = find_col(df_raw.columns.tolist(), "Created")
col_proj  = find_col(df_raw.columns.tolist(), "Project name") or find_col(df_raw.columns.tolist(), "Project")
col_labels= find_col(df_raw.columns.tolist(), "Labels")
col_comp  = find_col(df_raw.columns.tolist(), "Components")

missing = []
for name, col in [("Issue key", col_key), ("Summary", col_sum), ("Created", col_created), ("Project/Project name", col_proj)]:
    if not col: missing.append(name)
if missing:
    st.error("Eksik sütun(lar): " + ", ".join(missing))
    st.stop()

st.success(f"Yüklendi: {len(df_raw)} satır, {len(df_raw.columns)} sütun")

# ---- Kontrol paneli
c1, c2, c3 = st.columns([1,1,1])
with c1:
    sample_n = st.number_input("Rastgele örnek sayısı", min_value=1, max_value=max(1, len(df_raw)), value=min(10, len(df_raw)), step=1)
with c2:
    rng_seed = st.number_input("Rastgele tohum (seed)", min_value=0, max_value=10000, value=42, step=1)
with c3:
    default_checked_env = st.selectbox("Bu koşuda kontrol edilen ortam (default)", ["Mobile","Web","Backend","Unknown"], index=1)

# Ortam çıkarımı + örneklem
env_series = df_raw.apply(lambda r: infer_environment(r.get(col_proj), r.get(col_labels), r.get(col_comp)), axis=1)
work = df_raw.copy()
work.insert(2, "Environment", env_series)
work.insert(3, "Checked On", default_checked_env)

# Rastgele seç
if len(work) > sample_n:
    work = work.sample(n=int(sample_n), random_state=int(rng_seed))

# Kullanıcıya ortam düzeltmeleri için editör
st.subheader("Örneklem ve Ortam Düzeltmeleri")
st.caption("Gerekiyorsa Environment/Checked On kolonlarını değiştirin. Sonraki adımda değerlendirme yapılır.")

edited = st.data_editor(
    work[[col_key, col_sum, col_proj, "Environment", "Checked On", col_created, col_steps]].rename(columns={
        col_key: "Issue key", col_sum: "Summary", col_proj: "Project", col_created: "Created", col_steps: "Manual Test Steps"
    }),
    column_config={
        "Environment": st.column_config.SelectboxColumn(options=["Mobile","Web","Backend","Unknown"], help="Case'in ait olduğu ortam"),
        "Checked On": st.column_config.SelectboxColumn(options=["Mobile","Web","Backend","Unknown"], help="Bu koşuda kontrolü yaptığınız ortam"),
        "Manual Test Steps": st.column_config.TextColumn(help="Opsiyonel. JSON ham içerik veya boş.")
    },
    use_container_width=True,
    num_rows="fixed"
)

# Değerlendirme
results: List[Dict[str, Any]] = []
for _, row in edited.iterrows():
    decision, reason = evaluate_case(row.get("Summary"), row.get("Manual Test Steps"), row.get("Created"))
    results.append({
        "Issue key": row.get("Issue key"),
        "Summary": row.get("Summary"),
        "Project": row.get("Project"),
        "Environment": row.get("Environment"),
        "Checked On": row.get("Checked On"),
        "Created": row.get("Created"),
        "Evaluation": decision,
        "Reason": reason,
    })

out = pd.DataFrame(results)

# KPI'lar
k1, k2, k3, k4 = st.columns(4)
with k1: st.metric("Örneklem", len(out))
with k2: st.metric("Evet", int((out["Evaluation"] == "Evet").sum()))
with k3: st.metric("Evet*", int((out["Evaluation"] == "Evet*").sum()))
with k4: st.metric("Hayır", int((out["Evaluation"] == "Hayır").sum()))

st.subheader("Değerlendirme Sonuçları")
st.dataframe(out, use_container_width=True)

st.download_button(
    label="CSV indir (UTF-8 BOM, ;)",
    data=df_to_csv_bom(out, sep=";"),
    file_name="case_evaluation_sample.csv",
    mime="text/csv",
)

st.caption("Not: Heuristik kararlar yönlendiricidir; kesin doğrulama için ilgili ortamdaki gerçek uygulamada manuel test önerilir.")

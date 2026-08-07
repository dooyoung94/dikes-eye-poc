import os

import pandas as pd
import streamlit as st

from src.eda import build_eda
from src.naver_client import collect_visible_evidence, local_search
from src.normalize import normalize_evidence
from src.rfm import build_rfm

st.set_page_config(page_title="Dike's Eye POC", page_icon="⚖️", layout="centered")


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, os.getenv(name, default))
        return str(value) if value is not None else default
    except Exception:
        return os.getenv(name, default)


def naver_credentials() -> dict:
    return {
        "hub_id": secret("NAVER_API_HUB_CLIENT_ID"),
        "hub_secret": secret("NAVER_API_HUB_CLIENT_SECRET"),
        "legacy_id": secret("NAVER_LEGACY_CLIENT_ID"),
        "legacy_secret": secret("NAVER_LEGACY_CLIENT_SECRET"),
    }


st.markdown("""
<style>
.block-container {max-width: 900px; padding-top: 2rem; padding-bottom: 4rem;}
.hero {border:1px solid rgba(128,128,128,.2); border-radius:22px; padding:1.2rem 1.4rem; margin-bottom:1rem;}
.soft {opacity:.7;}
.card {border:1px solid rgba(128,128,128,.18); border-radius:16px; padding:.9rem 1rem; margin:.5rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="soft">Bias-aware Review Decision Agent · POC Step 3</div>
  <h1>⚖️ Dike's Eye</h1>
  <div>질문 → 식당 확인 → NAVER Evidence → EDA → RFM까지 검증합니다.</div>
</div>
""", unsafe_allow_html=True)

for key, default in {
    "messages": [("assistant", "지역과 식당명을 입력해 주세요.")],
    "query": "",
    "candidates": [],
    "search_status": "",
    "selected_place": None,
    "analysis": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

for role, text in st.session_state.messages:
    label = "⚖️ Dike's Eye" if role == "assistant" else "🙂 나"
    st.markdown(f"**{label}**  \n{text}")

with st.form("question_form", clear_on_submit=True):
    question = st.text_input("질문", placeholder="예: 성수 어니언", label_visibility="collapsed")
    submitted = st.form_submit_button("질문 보내기", type="primary", use_container_width=True)

if submitted and question.strip():
    text = question.strip()
    st.session_state.query = text
    st.session_state.candidates = []
    st.session_state.selected_place = None
    st.session_state.analysis = None
    st.session_state.search_status = ""
    st.session_state.messages.append(("user", text))
    st.session_state.messages.append(("assistant", "입력을 받았어요. 아래에서 식당 후보를 확인해 주세요."))
    st.success("질문 입력 정상")

if st.session_state.query:
    st.divider()
    st.subheader("📍 식당 후보 확인")
    st.write("검색어:", st.session_state.query)

    if st.button("NAVER 식당 후보 찾기", use_container_width=True):
        with st.spinner("NAVER Local 검색 중..."):
            candidates, status = local_search(st.session_state.query, **naver_credentials())
        st.session_state.search_status = status
        st.session_state.candidates = candidates or [{
            "title": st.session_state.query,
            "category": "직접 입력",
            "address": "",
            "link": "",
            "fallback": True,
        }]

    if st.session_state.search_status:
        if st.session_state.search_status == "정상":
            st.success("NAVER Local 검색 정상")
        else:
            st.warning(f"NAVER 검색 상태: {st.session_state.search_status}")

    if st.session_state.candidates:
        labels = [f"{x.get('title','')} · {x.get('category','')} · {x.get('address','')}" for x in st.session_state.candidates]
        idx = st.radio("이 식당이 맞나요?", range(len(labels)), format_func=lambda i: labels[i])
        chosen = st.session_state.candidates[idx]
        st.markdown(f"<div class='card'><b>📍 {chosen.get('title','')}</b><br><span class='soft'>{chosen.get('category','')}<br>{chosen.get('address','')}</span></div>", unsafe_allow_html=True)

        if st.button("네, 이 식당이 맞아요", type="primary", use_container_width=True):
            st.session_state.selected_place = chosen
            st.success(f"장소 확인 정상: {chosen.get('title','')}")

if st.session_state.selected_place:
    st.divider()
    st.subheader("📊 EDA / RFM 분석")
    place = st.session_state.selected_place

    c1, c2 = st.columns(2)
    with c1:
        day = st.text_input("방문 요일/날짜", placeholder="예: 토요일")
        time = st.text_input("방문 시간", placeholder="예: 19시")
    with c2:
        purpose = st.text_input("목적", placeholder="예: 소개팅")
        preference = st.text_input("중요 조건", placeholder="예: 조용함, 웨이팅")

    if st.button("Evidence 수집 + EDA/RFM 실행", type="primary", use_container_width=True):
        context = {"date_or_day": day, "time": time, "purpose": purpose, "preference": preference}
        with st.spinner("NAVER Blog/Cafe/Web Evidence 수집 중..."):
            raw = collect_visible_evidence(place.get("title", st.session_state.query), **naver_credentials())
        normalized = normalize_evidence(raw, context)
        rfm_rows, rfm_summary = build_rfm(normalized)
        eda = build_eda(rfm_rows)
        st.session_state.analysis = {"raw": raw, "rows": rfm_rows, "rfm": rfm_summary, "eda": eda, "context": context}

if st.session_state.analysis:
    a = st.session_state.analysis
    st.success(f"EDA/RFM 정상 · Evidence {len(a['rows'])}건")
    m1, m2, m3 = st.columns(3)
    m1.metric("Evidence", len(a["rows"]))
    m2.metric("평균 Recency", a["rfm"].get("avg_R", 0))
    m3.metric("평균 Match", a["rfm"].get("avg_M", 0))

    with st.expander("📊 EDA 결과", expanded=True):
        st.json(a["eda"])

    with st.expander("🧮 RFM 상위 Evidence", expanded=True):
        df = pd.DataFrame(a["rows"])
        cols = [c for c in ["evidence_id", "source", "title", "aspects", "contexts", "sentiment", "R", "F", "M", "priority"] if c in df.columns]
        st.dataframe(df[cols].head(30) if cols else df.head(30), use_container_width=True, hide_index=True)
        st.caption("RFM은 여기서 Recency / Frequency / Match 기반 Evidence 우선순위 휴리스틱으로 사용합니다.")

st.divider()
st.caption("POC Step 3 · NAVER Evidence + EDA + RFM / OpenAI · RCA · Wald · 최종 추천점수 미사용")

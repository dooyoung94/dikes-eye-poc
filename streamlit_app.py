import os
import re

import requests
import streamlit as st

st.set_page_config(page_title="Dike's Eye POC", page_icon="⚖️", layout="centered")


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, os.getenv(name, default))
        return str(value) if value is not None else default
    except Exception:
        return os.getenv(name, default)


def clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", str(text or "")).strip()


def naver_local(query: str) -> tuple[list[dict], str]:
    hub_id = secret("NAVER_API_HUB_CLIENT_ID")
    hub_secret = secret("NAVER_API_HUB_CLIENT_SECRET")
    legacy_id = secret("NAVER_LEGACY_CLIENT_ID")
    legacy_secret = secret("NAVER_LEGACY_CLIENT_SECRET")

    if hub_id and hub_secret:
        url = "https://naverapihub.apigw.ntruss.com/search/v1/local"
        headers = {
            "X-NCP-APIGW-API-KEY-ID": hub_id,
            "X-NCP-APIGW-API-KEY": hub_secret,
        }
    elif legacy_id and legacy_secret:
        url = "https://openapi.naver.com/v1/search/local.json"
        headers = {
            "X-Naver-Client-Id": legacy_id,
            "X-Naver-Client-Secret": legacy_secret,
        }
    else:
        return [], "NAVER Secret 미설정"

    try:
        response = requests.get(
            url,
            headers=headers,
            params={"query": query, "display": 5, "sort": "comment"},
            timeout=8,
        )
        response.raise_for_status()
        items = response.json().get("items", [])
        rows = []
        for item in items:
            rows.append({
                "title": clean_html(item.get("title", "")),
                "category": clean_html(item.get("category", "")),
                "address": clean_html(item.get("roadAddress") or item.get("address", "")),
                "link": str(item.get("link") or "").strip(),
            })
        return rows, "정상"
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


st.markdown(
    """
    <style>
    .block-container {max-width: 860px; padding-top: 2rem; padding-bottom: 4rem;}
    .hero {border:1px solid rgba(128,128,128,.2); border-radius:22px; padding:1.2rem 1.4rem; margin-bottom:1rem;}
    .soft {opacity:.7;}
    .card {border:1px solid rgba(128,128,128,.18); border-radius:16px; padding:.9rem 1rem; margin:.5rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="soft">Bias-aware Review Decision Agent · POC Step 2</div>
      <h1>⚖️ Dike's Eye</h1>
      <div>질문 입력 → 식당 후보 검색 → 장소 확인까지만 테스트합니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

for key, default in {
    "messages": [("assistant", "어디를 갈지 고민 중인가요? 지역과 식당명을 입력해 주세요.")],
    "query": "",
    "candidates": [],
    "search_status": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

for role, text in st.session_state.messages:
    label = "⚖️ Dike's Eye" if role == "assistant" else "🙂 나"
    st.markdown(f"**{label}**  \n{text}")

with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "질문",
        placeholder="예: 성수 어니언",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("질문 보내기", type="primary", use_container_width=True)

if submitted and question.strip():
    text = question.strip()
    st.session_state.query = text
    st.session_state.candidates = []
    st.session_state.search_status = ""
    st.session_state.messages.append(("user", text))
    st.session_state.messages.append(("assistant", "입력을 받았어요. 아래 버튼을 눌러 NAVER에서 식당 후보만 확인해 볼게요."))
    st.success("질문 입력 정상")

if st.session_state.query:
    st.divider()
    st.subheader("📍 식당 후보 확인")
    st.write("검색어:", st.session_state.query)

    if st.button("NAVER 식당 후보 찾기", use_container_width=True):
        with st.spinner("NAVER Local 검색 중..."):
            candidates, status = naver_local(st.session_state.query)
        st.session_state.search_status = status
        if candidates:
            st.session_state.candidates = candidates
        else:
            st.session_state.candidates = [{
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
        labels = [
            f"{x.get('title','')} · {x.get('category','')} · {x.get('address','')}"
            for x in st.session_state.candidates
        ]
        idx = st.radio("이 식당이 맞나요?", range(len(labels)), format_func=lambda i: labels[i])
        chosen = st.session_state.candidates[idx]
        st.markdown(
            f"<div class='card'><b>📍 {chosen.get('title','')}</b><br>"
            f"<span class='soft'>{chosen.get('category','')}<br>{chosen.get('address','')}</span></div>",
            unsafe_allow_html=True,
        )
        if st.button("네, 이 식당이 맞아요", type="primary", use_container_width=True):
            st.success(f"장소 확인 정상: {chosen.get('title','')}")
            st.info("POC Step 2 성공. 다음 단계에서 EDA/RFM/RCA 분석 엔진을 연결할 수 있습니다.")

st.divider()
st.caption("POC Step 2 · NAVER Local만 사용 / OpenAI · EDA · RFM · RCA · Wald 미사용")

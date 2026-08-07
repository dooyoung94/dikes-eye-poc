import os

import pandas as pd
import streamlit as st

from src.eda import build_eda
from src.llm_explain import generate_explanation
from src.naver_client import collect_hidden_evidence, collect_visible_evidence, local_search
from src.normalize import normalize_evidence
from src.rca import derive_rca
from src.rashomon import build_rashomon
from src.rfm import build_rfm
from src.scoring import score_decision
from src.wald import analyze_wald

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
.go {border-left:5px solid #21a366;}
.conditional {border-left:5px solid #d99a00;}
.avoid {border-left:5px solid #d64545;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="soft">Bias-aware Review Decision Agent · POC Step 6</div>
  <h1>⚖️ Dike's Eye</h1>
  <div>EDA → RFM → Rashomon → RCA → Wald → Dike Score를 계산하고, OpenAI는 선택적으로 결과를 설명만 합니다.</div>
</div>
""", unsafe_allow_html=True)

for key, default in {
    "messages": [("assistant", "지역과 식당명을 입력해 주세요.")],
    "query": "",
    "candidates": [],
    "search_status": "",
    "selected_place": None,
    "analysis": None,
    "explanation": None,
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
    st.session_state.explanation = None
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
            st.session_state.analysis = None
            st.session_state.explanation = None
            st.success(f"장소 확인 정상: {chosen.get('title','')}")

if st.session_state.selected_place:
    st.divider()
    st.subheader("📊 Evidence 분석")
    place = st.session_state.selected_place

    c1, c2 = st.columns(2)
    with c1:
        day = st.text_input("방문 요일/날짜", placeholder="예: 토요일")
        time = st.text_input("방문 시간", placeholder="예: 19시")
    with c2:
        purpose = st.text_input("목적", placeholder="예: 소개팅")
        preference = st.text_input("중요 조건", placeholder="예: 조용함, 웨이팅")

    if st.button("전체 Evidence 분석 실행", type="primary", use_container_width=True):
        context = {"date_or_day": day, "time": time, "purpose": purpose, "preference": preference}
        target = place.get("title", st.session_state.query)
        with st.spinner("Visible/Hidden Evidence 수집 및 분석 중..."):
            visible_raw = collect_visible_evidence(target, **naver_credentials())
            hidden_raw = collect_hidden_evidence(target, **naver_credentials())

            visible_norm = normalize_evidence(visible_raw, context)
            hidden_norm = normalize_evidence(hidden_raw, context)

            rfm_rows, rfm_summary = build_rfm(visible_norm)
            eda = build_eda(rfm_rows)
            rca = derive_rca(rfm_rows, context)
            rashomon = build_rashomon(rca)
            wald = analyze_wald(hidden_norm)
            decision = score_decision(rfm_rows, eda, rfm_summary, rca, wald)

        st.session_state.analysis = {
            "visible_raw": visible_raw,
            "hidden_raw": hidden_raw,
            "rows": rfm_rows,
            "hidden_rows": hidden_norm,
            "rfm": rfm_summary,
            "eda": eda,
            "context": context,
            "rca": rca,
            "rashomon": rashomon,
            "wald": wald,
            "decision": decision,
        }
        st.session_state.explanation = None

if st.session_state.analysis:
    a = st.session_state.analysis
    d = a["decision"]
    verdict = d.get("verdict", "CONDITIONAL")
    label = {"GO": "추천", "CONDITIONAL": "조건부 추천", "AVOID": "비추천"}.get(verdict, verdict)
    css = {"GO": "go", "CONDITIONAL": "conditional", "AVOID": "avoid"}.get(verdict, "conditional")

    st.success(f"Step 6 정상 · Visible {len(a['rows'])}건 / Hidden {len(a['hidden_rows'])}건")
    st.markdown(
        f"<div class='card {css}'><div class='soft'>Dike's Eye Verdict</div>"
        f"<h2 style='margin:.1rem 0'>{label} · {d.get('fit_score', 0)}/100</h2>"
        f"<div>판단 신뢰도 {d.get('confidence', 0)}%</div></div>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Visible", len(a["rows"]))
    m2.metric("Hidden", len(a["hidden_rows"]))
    m3.metric("RCA Risk", d.get("components", {}).get("rca_risk", 0))
    m4.metric("Wald Risk", d.get("components", {}).get("wald_risk", 0))

    if d.get("score_caps"):
        st.caption("점수 상한/보정: " + " · ".join(d["score_caps"]))

    st.divider()
    st.subheader("🤖 AI 설명")
    st.caption("점수와 verdict는 이미 확정되어 있으며, AI는 설명 문장만 생성합니다.")
    openai_ready = bool(secret("OPENAI_API_KEY"))
    st.write("OpenAI 상태:", "✅ Secret 연결" if openai_ready else "⚪ 미설정 · fallback 설명 사용")

    if st.button("AI 설명 생성", use_container_width=True):
        with st.spinner("분석 결과를 설명 문장으로 정리 중..."):
            st.session_state.explanation = generate_explanation(
                a,
                api_key=secret("OPENAI_API_KEY"),
                model=secret("OPENAI_MODEL", "gpt-5-mini"),
            )

    if st.session_state.explanation:
        e = st.session_state.explanation
        st.markdown(f"### {e.get('headline','')}")
        st.write(e.get("answer", ""))
        if e.get("reasons"):
            st.markdown("**근거**")
            for reason in e.get("reasons", []):
                st.markdown(f"- {reason}")
        if e.get("risks"):
            st.markdown("**주의할 점**")
            for risk in e.get("risks", []):
                st.markdown(f"- {risk}")
        st.caption("설명 소스: " + ("OpenAI" if e.get("source") == "openai" else "deterministic fallback"))

    with st.expander("⚖️ Dike Score 계산 근거", expanded=True):
        st.json(d)

    with st.expander("📊 EDA 결과"):
        st.json(a["eda"])

    with st.expander("🧮 RFM 상위 Evidence"):
        df = pd.DataFrame(a["rows"])
        cols = [c for c in ["evidence_id", "source", "title", "aspects", "contexts", "sentiment", "R", "F", "M", "priority"] if c in df.columns]
        st.dataframe(df[cols].head(30) if cols else df.head(30), use_container_width=True, hide_index=True)
        st.caption("RFM은 Recency / Frequency / Match 기반 Evidence 우선순위 휴리스틱입니다.")

    with st.expander("🎭 Rashomon · 서로 다른 진실"):
        st.write(a["rashomon"].get("summary", ""))
        st.json(a["rashomon"])

    with st.expander("🧩 RCA · 왜 의견이 갈렸나"):
        st.caption(a["rca"].get("interpretation", ""))
        st.write("사용자 조건 flags:", a["rca"].get("user_context_flags", []))
        st.write("사용자 조건 정렬 위험도:", a["rca"].get("aligned_risk", 0))
        candidates_df = pd.DataFrame(a["rca"].get("cause_candidates", []))
        if candidates_df.empty:
            st.info("현재 Evidence에서는 조건별 차이를 설명할 만큼 충분한 RCA 후보가 없습니다.")
        else:
            st.dataframe(candidates_df, use_container_width=True, hide_index=True)

    with st.expander("🕳️ Wald · 사라진 진실", expanded=True):
        st.caption(a["wald"].get("interpretation", ""))
        st.json(a["wald"])
        hidden_df = pd.DataFrame(a["hidden_rows"])
        if not hidden_df.empty:
            hidden_cols = [c for c in ["evidence_id", "source", "query", "title", "snippet"] if c in hidden_df.columns]
            st.dataframe(hidden_df[hidden_cols].head(30), use_container_width=True, hide_index=True)

st.divider()
st.caption("POC Step 6 · deterministic Dike Score + optional OpenAI explanation. OpenAI는 점수/판단을 변경하지 않습니다.")

import os

import pandas as pd
import streamlit as st

from src.eda import build_eda
from src.intent import parse_intent
from src.llm_explain import generate_explanation
from src.naver_client import collect_hidden_evidence, collect_visible_evidence, local_search
from src.normalize import normalize_evidence
from src.rca import derive_rca
from src.rashomon import build_rashomon
from src.reporting import build_user_report
from src.rfm import build_rfm
from src.scoring import score_decision
from src.wald import analyze_wald

st.set_page_config(
    page_title="Dike's Eye",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


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


def reset_after_question(intent: dict) -> None:
    st.session_state.intent = intent
    st.session_state.candidates = []
    st.session_state.search_status = ""
    st.session_state.selected_target = None
    st.session_state.analysis = None
    st.session_state.user_report = None
    st.session_state.explanation = None
    st.session_state.last_error = ""
    st.session_state.ctx_day = intent.get("date_or_day", "")
    st.session_state.ctx_time = intent.get("time", "")
    st.session_state.ctx_purpose = intent.get("purpose", "")
    st.session_state.ctx_preference = intent.get("preference", "")
    st.session_state.target_edit = intent.get("target", "")


def run_decision(target: str, kind: str, context: dict) -> dict:
    creds = naver_credentials()
    visible_raw = collect_visible_evidence(target, kind=kind, **creds)
    hidden_raw = collect_hidden_evidence(target, kind=kind, **creds)

    visible_norm = normalize_evidence(visible_raw, context)
    hidden_norm = normalize_evidence(hidden_raw, context)

    rfm_rows, rfm_summary = build_rfm(visible_norm)
    eda = build_eda(rfm_rows)
    rca = derive_rca(rfm_rows, context)
    rashomon = build_rashomon(rca)
    wald = analyze_wald(hidden_norm, kind=kind)
    decision = score_decision(rfm_rows, eda, rfm_summary, rca, wald)

    return {
        "kind": kind,
        "target": target,
        "context": context,
        "visible_raw": visible_raw,
        "hidden_raw": hidden_raw,
        "rows": rfm_rows,
        "hidden_rows": hidden_norm,
        "rfm": rfm_summary,
        "eda": eda,
        "rca": rca,
        "rashomon": rashomon,
        "wald": wald,
        "decision": decision,
    }


st.markdown(
    """
<style>
.block-container {max-width: 860px; padding-top: 1.8rem; padding-bottom: 5rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.hero-title {font-size: 2.35rem; font-weight: 800; letter-spacing: -0.04em; margin-bottom: .25rem;}
.hero-sub {font-size: 1.05rem; opacity: .74; line-height: 1.65; margin-bottom: 1.15rem;}
.eyebrow {font-size: .76rem; font-weight: 800; letter-spacing: .08em; opacity: .55; text-transform: uppercase;}
.soft {opacity: .7;}
.small {font-size: .88rem; opacity: .68;}
.stButton > button {border-radius: 999px; min-height: 2.7rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Bias-aware decision agent</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">⚖️ Dike\'s Eye</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">평균 리뷰를 요약하는 대신, <b>왜 의견이 갈리는지</b>와 '
    '<b>리뷰에 잘 남지 않는 이탈 신호</b>를 함께 보고 <b>내 조건에서 선택해도 되는지</b> 판단합니다.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("**이렇게 물어보세요**")
    st.caption("식당 · `토요일 7시 소개팅인데 성수 어니언 어때?`")
    st.caption("상품 · `출퇴근용으로 소니 WH-1000XM6 사도 될까? 배터리랑 착용감이 중요해`")

DEFAULTS = {
    "intent": None,
    "candidates": [],
    "search_status": "",
    "selected_target": None,
    "analysis": None,
    "user_report": None,
    "explanation": None,
    "last_error": "",
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "질문",
        placeholder="어디를 갈지, 무엇을 살지 상황까지 같이 말해 주세요.",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Dike's Eye에게 물어보기", type="primary", use_container_width=True)

if submitted and question.strip():
    try:
        reset_after_question(parse_intent(question.strip()))
    except Exception as exc:
        st.session_state.last_error = f"{type(exc).__name__}: {exc}"

intent = st.session_state.intent

if intent:
    kind = intent.get("kind", "restaurant")
    kind_label = "식당" if kind == "restaurant" else "상품"
    st.divider()
    st.markdown("### 1. 제가 이렇게 이해했어요")

    with st.container(border=True):
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f"**{kind_label} · {intent.get('target', '')}**")
            context_bits = [
                intent.get("date_or_day", ""),
                intent.get("time", ""),
                intent.get("purpose", ""),
                intent.get("preference", ""),
            ]
            context_text = " · ".join(x for x in context_bits if x)
            st.caption(context_text or "추가 조건은 다음 단계에서 입력할 수 있어요.")
        with c2:
            st.metric("질문 해석", f"{int(intent.get('parse_confidence', 0) * 100)}%")

    if kind == "restaurant" and not st.session_state.selected_target:
        if st.button("NAVER에서 장소 확인", type="primary", use_container_width=True):
            try:
                with st.spinner("장소를 확인하고 있어요..."):
                    candidates, status = local_search(intent.get("target") or intent.get("original", ""), **naver_credentials())
                st.session_state.search_status = status
                st.session_state.candidates = candidates or [{
                    "title": intent.get("target") or intent.get("original", ""),
                    "category": "직접 입력",
                    "address": "",
                    "fallback": True,
                }]
            except Exception as exc:
                st.session_state.last_error = f"{type(exc).__name__}: {exc}"

        if st.session_state.candidates:
            labels = [
                f"{x.get('title','')} · {x.get('category','')} · {x.get('address','')}"
                for x in st.session_state.candidates
            ]
            idx = st.radio(
                "이 장소가 맞나요?",
                range(len(labels)),
                format_func=lambda i: labels[i],
            )
            chosen = st.session_state.candidates[idx]
            if st.button("이 장소로 결정 분석", type="primary", use_container_width=True):
                st.session_state.selected_target = {
                    "kind": "restaurant",
                    "name": chosen.get("title") or intent.get("target", ""),
                    "meta": chosen,
                }

    if kind == "product" and not st.session_state.selected_target:
        with st.form("product_confirm_form"):
            product_name = st.text_input(
                "제품명",
                key="target_edit",
                help="모델명까지 정확할수록 Evidence 검색 정확도가 높아집니다.",
            )
            product_confirmed = st.form_submit_button("이 제품으로 결정 분석", type="primary", use_container_width=True)
        if product_confirmed and product_name.strip():
            st.session_state.selected_target = {
                "kind": "product",
                "name": product_name.strip(),
                "meta": {},
            }

selected = st.session_state.selected_target

if selected and not st.session_state.analysis:
    st.divider()
    st.markdown("### 2. 내 조건만 확인할게요")
    st.caption("여기 조건이 Dike's Eye의 핵심입니다. 같은 리뷰라도 누구에게는 장점이고 누구에게는 단점일 수 있습니다.")

    with st.form("context_form"):
        if selected["kind"] == "restaurant":
            a, b = st.columns(2)
            with a:
                st.text_input("방문 날짜/요일", key="ctx_day", placeholder="예: 토요일")
                st.text_input("시간", key="ctx_time", placeholder="예: 19:00")
            with b:
                st.text_input("목적", key="ctx_purpose", placeholder="예: 소개팅, 데이트")
                st.text_input("중요한 조건", key="ctx_preference", placeholder="예: 조용함, 웨이팅")
        else:
            a, b = st.columns(2)
            with a:
                st.text_input("사용 목적", key="ctx_purpose", placeholder="예: 출퇴근, 업무, 게임")
                st.text_input("중요한 조건", key="ctx_preference", placeholder="예: 배터리, 착용감, 무게")
            with b:
                st.text_input("사용 시점/상황", key="ctx_day", placeholder="예: 매일, 주말 여행")
                st.text_input("추가 조건", key="ctx_time", placeholder="예: 하루 3시간 사용")

        analyze = st.form_submit_button("내 조건으로 판단하기", type="primary", use_container_width=True)

    if analyze:
        context = {
            "date_or_day": st.session_state.ctx_day,
            "time": st.session_state.ctx_time,
            "purpose": st.session_state.ctx_purpose,
            "preference": st.session_state.ctx_preference,
        }
        try:
            with st.spinner("보이는 리뷰와 놓치기 쉬운 이탈 신호를 함께 분석하고 있어요..."):
                analysis = run_decision(selected["name"], selected["kind"], context)
                report = build_user_report(analysis, selected["name"], selected["kind"])
            st.session_state.analysis = analysis
            st.session_state.user_report = report
            st.session_state.explanation = None
        except Exception as exc:
            st.session_state.last_error = f"{type(exc).__name__}: {exc}"

if st.session_state.analysis and st.session_state.user_report:
    a = st.session_state.analysis
    r = st.session_state.user_report
    d = a["decision"]

    st.divider()
    st.markdown("### 3. Dike's Eye의 판단")

    verdict = d.get("verdict", "CONDITIONAL")
    icon = {"GO": "✅", "CONDITIONAL": "🟡", "AVOID": "🔴"}.get(verdict, "🟡")
    with st.container(border=True):
        st.markdown(f"## {icon} {r['headline']}")
        st.write(r["summary"])
        m1, m2, m3 = st.columns(3)
        m1.metric("조건 적합도", f"{d.get('fit_score', 0)}/100")
        m2.metric("판단 신뢰도", f"{d.get('confidence', 0)}%")
        m3.metric("검토 Evidence", f"{len(a.get('rows', [])) + len(a.get('hidden_rows', []))}건")
        st.caption(r["confidence_note"])

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### 왜 이렇게 판단했나요?")
            for item in r.get("why", []):
                st.markdown(f"- {item}")
    with c2:
        with st.container(border=True):
            st.markdown("#### 무엇을 조심해야 하나요?")
            if r.get("risks"):
                for item in r["risks"]:
                    st.markdown(f"- {item}")
            else:
                st.write("현재 Evidence에서 강한 추가 위험은 확인되지 않았습니다.")

    with st.container(border=True):
        st.markdown("#### 그래서 어떻게 결정하면 되나요?")
        for item in r.get("actions", []):
            st.markdown(f"- **{item}**")

    st.caption(r["method_note"])

    with st.expander("AI에게 결과를 더 자연스럽게 설명시키기"):
        st.caption("점수와 추천 여부는 이미 확정되어 있습니다. AI는 분석 결과를 설명만 하며 판단을 바꾸지 않습니다.")
        if st.button("AI 설명 생성", use_container_width=True):
            with st.spinner("결과를 읽기 쉽게 정리하고 있어요..."):
                st.session_state.explanation = generate_explanation(
                    a,
                    api_key=secret("OPENAI_API_KEY"),
                    model=secret("OPENAI_MODEL", "gpt-5-mini"),
                )
        if st.session_state.explanation:
            e = st.session_state.explanation
            st.markdown(f"**{e.get('headline', '')}**")
            st.write(e.get("answer", ""))
            for reason in e.get("reasons", []):
                st.markdown(f"- {reason}")

    with st.expander("왜 이런 결론이 나왔는지 자세히 보기"):
        st.markdown("#### 🎭 서로 다른 진실 — Rashomon")
        st.write(a.get("rashomon", {}).get("summary", "충돌 패턴이 충분하지 않습니다."))

        st.markdown("#### 🧩 의견이 갈린 이유 — RCA")
        st.caption(a.get("rca", {}).get("interpretation", ""))
        rca_df = pd.DataFrame(a.get("rca", {}).get("cause_candidates", []))
        if not rca_df.empty:
            preferred_cols = [c for c in ["aspect", "context", "effect", "lift", "support_count", "confidence", "user_aligned"] if c in rca_df.columns]
            st.dataframe(rca_df[preferred_cols], use_container_width=True, hide_index=True)
        else:
            st.info("조건별 차이를 설명할 만큼 충분한 RCA 후보가 없습니다.")

        st.markdown("#### 🕳️ 리뷰 밖의 신호 — Wald")
        st.caption(a.get("wald", {}).get("interpretation", ""))
        signal_counts = a.get("wald", {}).get("signal_counts", {})
        st.write(signal_counts or "강한 이탈 신호가 확인되지 않았습니다.")

        st.markdown("#### 🧮 Evidence 우선순위")
        df = pd.DataFrame(a.get("rows", []))
        cols = [c for c in ["source", "title", "aspects", "contexts", "sentiment", "R", "F", "M", "priority"] if c in df.columns]
        if not df.empty:
            st.dataframe(df[cols].head(20), use_container_width=True, hide_index=True)
        st.caption("R=최신성 · F=반복성/출처다양성 · M=내 조건과의 일치도. 우선순위 = 0.35R + 0.25F + 0.40M")

        with st.expander("개발/검증용 원시 지표"):
            st.json({
                "eda": a.get("eda", {}),
                "rfm": a.get("rfm", {}),
                "decision": a.get("decision", {}),
            })

    if st.button("새로운 질문하기", use_container_width=True):
        for key in list(DEFAULTS) + ["ctx_day", "ctx_time", "ctx_purpose", "ctx_preference", "target_edit"]:
            st.session_state.pop(key, None)
        st.rerun()

if st.session_state.last_error:
    st.error("처리 중 문제가 발생했습니다. 입력한 질문과 이전 분석 결과는 안전하게 유지됩니다.")
    with st.expander("오류 정보"):
        st.code(st.session_state.last_error)

st.divider()
st.caption("Dike's Eye · 평균 리뷰가 아니라 내 조건에서의 선택을 돕는 Evidence 기반 Decision Agent")

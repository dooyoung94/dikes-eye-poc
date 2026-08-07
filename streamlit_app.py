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
:root {
    --ivory: #f7f3e9;
    --paper: #fffdf7;
    --navy: #172238;
    --navy-soft: #263750;
    --gold: #b58a3c;
    --gold-soft: #ddc994;
    --line: rgba(181, 138, 60, .30);
    --muted: #6c6a65;
}

[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at 14% 0%, rgba(181,138,60,.10), transparent 25%),
      radial-gradient(circle at 88% 4%, rgba(23,34,56,.055), transparent 24%),
      linear-gradient(180deg, #fbf8f0 0%, #f5f0e5 100%);
    color: var(--navy);
}

[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 930px; padding-top: 1.6rem; padding-bottom: 6rem;}

.dike-hero {
    position: relative;
    border: 1px solid var(--line);
    border-radius: 26px;
    padding: 1.65rem 1.75rem 1.55rem 1.75rem;
    background: linear-gradient(145deg, rgba(255,253,247,.94), rgba(248,242,226,.92));
    box-shadow: 0 18px 50px rgba(23,34,56,.07);
    overflow: hidden;
    margin-bottom: 1rem;
}
.dike-hero:after {
    content: "⚖";
    position: absolute;
    right: 1.1rem;
    top: -.6rem;
    font-size: 8.5rem;
    color: rgba(181,138,60,.085);
    transform: rotate(-7deg);
}
.dike-kicker {font-size:.72rem; letter-spacing:.18em; font-weight:800; color:var(--gold); text-transform:uppercase;}
.dike-title {font-family: Georgia, 'Times New Roman', serif; font-size:2.75rem; line-height:1; font-weight:700; color:var(--navy); margin:.45rem 0 .55rem 0; letter-spacing:-.035em;}
.dike-sub {max-width:690px; color:#4f5360; line-height:1.7; font-size:1.02rem;}
.dike-motto {margin-top:.8rem; color:var(--gold); font-family: Georgia, 'Times New Roman', serif; font-style:italic; font-size:.94rem;}

.seal-row {display:flex; gap:.55rem; flex-wrap:wrap; margin:.95rem 0 .15rem 0;}
.seal-chip {border:1px solid var(--line); border-radius:999px; padding:.32rem .65rem; background:rgba(255,255,255,.55); font-size:.78rem; color:var(--navy-soft);}

.section-kicker {font-size:.72rem; letter-spacing:.12em; font-weight:800; color:var(--gold); text-transform:uppercase; margin-bottom:.18rem;}
.opinion-title {font-family: Georgia, 'Times New Roman', serif; font-size:1.65rem; font-weight:700; color:var(--navy); margin-bottom:.2rem;}
.opinion-rule {height:1px; background:linear-gradient(90deg,var(--gold),rgba(181,138,60,0)); margin:.65rem 0 1rem 0;}

.opinion-sheet {
    border:1px solid var(--line);
    border-radius:20px;
    padding:1.25rem 1.35rem;
    background:rgba(255,253,247,.88);
    box-shadow:0 12px 34px rgba(23,34,56,.05);
}
.meta-grid {display:grid; grid-template-columns:110px 1fr; gap:.35rem .8rem; font-size:.9rem; margin:.7rem 0;}
.meta-label {color:#8b754d; font-weight:700;}
.meta-value {color:var(--navy);}
.verdict-banner {border-left:5px solid var(--gold); padding:.85rem 1rem; background:rgba(181,138,60,.075); border-radius:0 14px 14px 0; margin:.85rem 0;}
.verdict-banner strong {font-family: Georgia, 'Times New Roman', serif; font-size:1.2rem; color:var(--navy);}

.brief-card {border:1px solid rgba(23,34,56,.10); border-radius:16px; background:rgba(255,255,255,.50); padding:.9rem 1rem; min-height:100%;}
.brief-card h4 {font-family: Georgia, 'Times New Roman', serif; color:var(--navy); margin:.1rem 0 .55rem 0;}

.notice-box {border:1px dashed rgba(181,138,60,.48); border-radius:14px; padding:.75rem .9rem; background:rgba(181,138,60,.05); color:#5b584f; font-size:.88rem; line-height:1.55;}

[data-testid="stMetric"] {background:rgba(255,253,247,.76); border:1px solid rgba(181,138,60,.20); padding:.72rem .8rem; border-radius:14px;}
[data-testid="stMetricValue"] {font-family: Georgia, 'Times New Roman', serif; color:var(--navy); font-size:1.45rem;}

.stButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius:999px;
    min-height:2.8rem;
    border:1px solid rgba(181,138,60,.42);
}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background:linear-gradient(135deg,#1a2942,#243a5b);
    color:#fffaf0;
    border-color:#243a5b;
}
.stTextInput input {border-radius:13px !important; background:rgba(255,253,247,.84) !important;}
[data-testid="stExpander"] {border:1px solid rgba(181,138,60,.19); border-radius:14px; background:rgba(255,253,247,.50);}

@media (max-width: 650px) {
  .dike-title {font-size:2.2rem;}
  .dike-hero {padding:1.3rem 1.15rem;}
  .meta-grid {grid-template-columns:86px 1fr;}
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="dike-hero">
  <div class="dike-kicker">Dike · Goddess of Justice</div>
  <div class="dike-title">Dike's Eye</div>
  <div class="dike-sub">
    다수의 평가를 그대로 따르지 않습니다. 상반된 진술과 누락되기 쉬운 Evidence를 함께 검토하고,
    <b>귀하의 실제 조건에서 선택이 타당한지</b> 의견을 제시합니다.
  </div>
  <div class="seal-row">
    <span class="seal-chip">⚖ 상반된 의견 검토</span>
    <span class="seal-chip">◌ 누락 Evidence 탐색</span>
    <span class="seal-chip">§ 사용자 조건 적용</span>
  </div>
  <div class="dike-motto">Audiatur et altera pars — 다른 쪽의 이야기 또한 들어야 한다.</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown('<div class="section-kicker">Request for Review</div>', unsafe_allow_html=True)
    st.markdown("#### 검토를 요청할 사안을 말씀해 주세요")
    st.caption("대상만 적기보다 시간·목적·중요 조건까지 함께 말씀해 주시면 보다 정밀하게 검토할 수 있습니다.")
    st.caption("예시 · `토요일 7시 소개팅인데 성수 어니언 어때?`")
    st.caption("예시 · `출퇴근용으로 소니 WH-1000XM6 사도 될까? 배터리랑 착용감이 중요해`")

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
        "검토 요청",
        placeholder="검토받고 싶은 선택과 상황을 한 문장으로 입력해 주세요.",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("검토 접수", type="primary", use_container_width=True)

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
    st.markdown('<div class="section-kicker">Preliminary Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="opinion-title">접수 내용 확인</div>', unsafe_allow_html=True)
    st.markdown('<div class="opinion-rule"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        c1, c2 = st.columns([2.4, 1])
        with c1:
            st.markdown(f"**검토 대상 · {intent.get('target', '')}**")
            context_bits = [
                intent.get("date_or_day", ""),
                intent.get("time", ""),
                intent.get("purpose", ""),
                intent.get("preference", ""),
            ]
            context_text = " · ".join(x for x in context_bits if x)
            st.caption(f"분류: {kind_label}")
            st.caption("확인된 조건: " + (context_text or "추가 조건 없음"))
        with c2:
            st.metric("질문 해석", f"{int(intent.get('parse_confidence', 0) * 100)}%")

    if kind == "restaurant" and not st.session_state.selected_target:
        st.caption("식당 사안은 동명이인 또는 지점 오인 가능성을 줄이기 위해 장소 확인 절차를 거칩니다.")
        if st.button("NAVER에서 검토 대상 확인", type="primary", use_container_width=True):
            try:
                with st.spinner("검토 대상을 확인하고 있습니다..."):
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
            idx = st.radio("검토할 장소를 선택해 주세요.", range(len(labels)), format_func=lambda i: labels[i])
            chosen = st.session_state.candidates[idx]
            if st.button("이 장소를 검토 대상으로 확정", type="primary", use_container_width=True):
                st.session_state.selected_target = {
                    "kind": "restaurant",
                    "name": chosen.get("title") or intent.get("target", ""),
                    "meta": chosen,
                }

    if kind == "product" and not st.session_state.selected_target:
        st.caption("상품은 모델명이 정확할수록 관련 Evidence와 다른 모델의 후기가 섞일 가능성이 낮아집니다.")
        with st.form("product_confirm_form"):
            product_name = st.text_input("제품명 / 모델명", key="target_edit")
            product_confirmed = st.form_submit_button("이 제품을 검토 대상으로 확정", type="primary", use_container_width=True)
        if product_confirmed and product_name.strip():
            st.session_state.selected_target = {"kind": "product", "name": product_name.strip(), "meta": {}}

selected = st.session_state.selected_target

if selected and not st.session_state.analysis:
    st.divider()
    st.markdown('<div class="section-kicker">Scope of Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="opinion-title">검토 범위와 조건</div>', unsafe_allow_html=True)
    st.markdown('<div class="opinion-rule"></div>', unsafe_allow_html=True)
    st.write(
        "동일한 대상도 이용 시점·목적·중요 조건에 따라 결론이 달라질 수 있습니다. "
        "아래 조건은 단순 참고정보가 아니라 Evidence의 우선순위를 결정하는 핵심 기준으로 사용됩니다."
    )

    with st.form("context_form"):
        if selected["kind"] == "restaurant":
            a, b = st.columns(2)
            with a:
                st.text_input("방문 날짜/요일", key="ctx_day", placeholder="예: 토요일")
                st.text_input("예정 시간", key="ctx_time", placeholder="예: 19:00")
            with b:
                st.text_input("방문 목적", key="ctx_purpose", placeholder="예: 소개팅, 데이트")
                st.text_input("중요하게 보는 조건", key="ctx_preference", placeholder="예: 조용함, 웨이팅, 주차")
        else:
            a, b = st.columns(2)
            with a:
                st.text_input("사용 목적", key="ctx_purpose", placeholder="예: 출퇴근, 업무, 게임")
                st.text_input("중요하게 보는 조건", key="ctx_preference", placeholder="예: 배터리, 착용감, 무게")
            with b:
                st.text_input("사용 시점/상황", key="ctx_day", placeholder="예: 매일, 주말 여행")
                st.text_input("추가 조건", key="ctx_time", placeholder="예: 하루 3시간 사용")

        analyze = st.form_submit_button("Evidence 검토 시작", type="primary", use_container_width=True)

    if analyze:
        context = {
            "date_or_day": st.session_state.ctx_day,
            "time": st.session_state.ctx_time,
            "purpose": st.session_state.ctx_purpose,
            "preference": st.session_state.ctx_preference,
        }
        try:
            with st.spinner("찬성·반대 Evidence와 누락 가능성이 있는 신호를 함께 검토하고 있습니다..."):
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
    verdict = d.get("verdict", "CONDITIONAL")
    verdict_mark = {"GO": "권고", "CONDITIONAL": "조건부 권고", "AVOID": "권고 유보"}.get(verdict, "조건부 권고")

    st.divider()
    st.markdown('<div class="section-kicker">Written Opinion</div>', unsafe_allow_html=True)
    st.markdown('<div class="opinion-title">DIKE 검토 의견서</div>', unsafe_allow_html=True)
    st.markdown('<div class="opinion-rule"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            f"""
<div class="meta-grid">
  <div class="meta-label">사안</div><div class="meta-value">{r.get('matter','')}</div>
  <div class="meta-label">검토 범위</div><div class="meta-value">{r.get('scope','')}</div>
  <div class="meta-label">검토 방식</div><div class="meta-value">공개 Evidence의 상반된 진술·조건 차이·누락 신호 종합</div>
</div>
<div class="verdict-banner"><strong>검토 결론 · {verdict_mark}</strong><br>{r.get('summary','')}</div>
""",
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("조건 적합도", f"{d.get('fit_score', 0)}/100")
        m2.metric("판단 신뢰도", f"{d.get('confidence', 0)}%")
        m3.metric("검토 Evidence", f"{len(a.get('rows', [])) + len(a.get('hidden_rows', []))}건")
        st.markdown(f"<div class='notice-box'>{r.get('confidence_note','')}</div>", unsafe_allow_html=True)

    st.markdown("#### I. 확인된 사실")
    for item in r.get("findings", []):
        st.markdown(f"- {item}")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("#### II. 상반된 Evidence")
            if r.get("conflicting_evidence"):
                for item in r["conflicting_evidence"]:
                    st.markdown(f"- {item}")
            else:
                st.write("현재 검토 범위에서 강한 반대 Evidence는 두드러지지 않았습니다.")
    with c2:
        with st.container(border=True):
            st.markdown("#### III. 리뷰에 나타나지 않을 수 있는 사실")
            for item in r.get("missing_side", []):
                st.markdown(f"- {item}")

    with st.container(border=True):
        st.markdown("#### IV. 판단의 한계")
        for item in r.get("limitations", []):
            st.markdown(f"- {item}")

    with st.container(border=True):
        st.markdown("#### V. 권고 의견")
        for item in r.get("recommendations", []):
            st.markdown(f"- **{item}**")

    st.caption(r.get("method_note", ""))
    st.caption(r.get("closing_note", ""))

    with st.expander("⚖️ 검토 의견을 더 상세한 문장으로 정리"):
        st.caption("AI는 이미 확정된 점수와 권고 여부를 변경하지 않습니다. 구조화된 검토 결과를 보다 읽기 쉬운 의견서 문장으로만 정리합니다.")
        if st.button("상세 의견 작성", use_container_width=True):
            with st.spinner("검토 기록을 정리하고 있습니다..."):
                st.session_state.explanation = generate_explanation(
                    a,
                    api_key=secret("OPENAI_API_KEY"),
                    model=secret("OPENAI_MODEL", "gpt-5-mini"),
                )
        if st.session_state.explanation:
            e = st.session_state.explanation
            st.markdown(f"### {e.get('headline', '')}")
            st.write(e.get("answer", ""))
            st.markdown("**주요 검토 근거**")
            for reason in e.get("reasons", []):
                st.markdown(f"- {reason}")
            if e.get("risks"):
                st.markdown("**유의사항**")
                for risk in e.get("risks", []):
                    st.markdown(f"- {risk}")

    with st.expander("🔎 Evidence 검토 기록 보기"):
        st.markdown("##### A. 서로 다른 진술 — Rashomon")
        st.write(a.get("rashomon", {}).get("summary", "충돌 패턴이 충분하지 않습니다."))

        st.markdown("##### B. 조건별 차이 — RCA")
        st.caption(a.get("rca", {}).get("interpretation", ""))
        rca_df = pd.DataFrame(a.get("rca", {}).get("cause_candidates", []))
        if not rca_df.empty:
            preferred_cols = [c for c in ["aspect", "context", "effect", "lift", "support_count", "confidence", "user_aligned"] if c in rca_df.columns]
            st.dataframe(rca_df[preferred_cols], use_container_width=True, hide_index=True)
        else:
            st.info("조건별 차이를 설명할 만큼 충분한 RCA 후보가 없습니다.")

        st.markdown("##### C. 누락 가능성 — Wald")
        st.caption(a.get("wald", {}).get("interpretation", ""))
        st.write(a.get("wald", {}).get("signal_counts", {}) or "강한 이탈 신호가 확인되지 않았습니다.")

        st.markdown("##### D. 우선 검토 Evidence")
        df = pd.DataFrame(a.get("rows", []))
        cols = [c for c in ["source", "title", "aspects", "contexts", "sentiment", "R", "F", "M", "priority"] if c in df.columns]
        if not df.empty:
            st.dataframe(df[cols].head(20), use_container_width=True, hide_index=True)
        st.caption("R=최신성 · F=반복성/출처 다양성 · M=사용자 조건 일치도 · Evidence Priority = 0.35R + 0.25F + 0.40M")

        with st.expander("개발·검증용 원시 지표"):
            st.json({"eda": a.get("eda", {}), "rfm": a.get("rfm", {}), "decision": a.get("decision", {})})

    if st.button("새 사안 검토하기", use_container_width=True):
        for key in list(DEFAULTS) + ["ctx_day", "ctx_time", "ctx_purpose", "ctx_preference", "target_edit"]:
            st.session_state.pop(key, None)
        st.rerun()

if st.session_state.last_error:
    st.error("검토 처리 중 문제가 발생했습니다. 기존 입력 내용은 유지됩니다.")
    with st.expander("오류 정보"):
        st.code(st.session_state.last_error)

st.divider()
st.markdown(
    "<div style='text-align:center;color:#86775a;font-size:.82rem;'>⚖ DIKE'S EYE · Evidence-based Decision Review</div>",
    unsafe_allow_html=True,
)

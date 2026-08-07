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
from src.reporting import ASPECT_LABELS, WALD_LABELS, build_user_report
from src.rfm import build_rfm
from src.scoring import score_decision
from src.wald import analyze_wald

st.set_page_config(page_title="Dike's Eye", page_icon="⚖️", layout="centered", initial_sidebar_state="collapsed")


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


def bar_row(label: str, value: float, count_text: str, tone: str = "gold") -> None:
    width = max(0, min(100, float(value)))
    cls = "bar-negative" if tone == "negative" else "bar-positive" if tone == "positive" else "bar-gold"
    st.markdown(
        f"""
        <div class="bar-wrap">
          <div class="bar-top"><span>{label}</span><strong>{count_text}</strong></div>
          <div class="bar-track"><div class="bar-fill {cls}" style="width:{width:.1f}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
<style>
:root {
  --bg:#f7f5f0; --card:#ffffff; --ink:#20242d; --sub:#6f7480; --gold:#aa8351;
  --gold2:#d9c3a3; --line:#ebe6dd; --green:#4e8d72; --red:#b65f63; --soft:#f2eee7;
}
html, body, [class*="css"] {font-family:Pretendard,-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Segoe UI",sans-serif;}
[data-testid="stAppViewContainer"] {background:linear-gradient(180deg,#faf9f6 0%,var(--bg) 100%); color:var(--ink);}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:920px; padding-top:1.4rem; padding-bottom:5rem;}
.hero {position:relative; overflow:hidden; background:linear-gradient(135deg,#242a35,#303947); color:white; border-radius:28px; padding:1.7rem 1.8rem; margin-bottom:1rem; box-shadow:0 18px 45px rgba(25,29,36,.12);}
.hero:after {content:"⚖"; position:absolute; right:1rem; top:-1.2rem; font-size:9rem; opacity:.08;}
.hero-eyebrow {font-size:.74rem; font-weight:800; letter-spacing:.16em; color:#dbc29b;}
.hero-title {font-size:2.45rem; font-weight:850; letter-spacing:-.05em; margin:.3rem 0 .45rem;}
.hero-sub {font-size:1rem; line-height:1.7; max-width:680px; color:#e4e7ec;}
.hero-chips {display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1rem;}
.hero-chip {font-size:.78rem;padding:.35rem .7rem;border:1px solid rgba(255,255,255,.18);border-radius:999px;background:rgba(255,255,255,.06);}
.section-title {font-size:1.28rem;font-weight:800;letter-spacing:-.03em;margin:.2rem 0 .25rem;}
.section-sub {font-size:.9rem;color:var(--sub);margin-bottom:.8rem;}
.result-hero {background:var(--card);border:1px solid var(--line);border-radius:22px;padding:1.2rem 1.25rem;box-shadow:0 10px 30px rgba(31,35,41,.06);}
.result-label {font-size:.74rem;font-weight:800;color:var(--gold);letter-spacing:.12em;text-transform:uppercase;}
.result-title {font-size:1.8rem;font-weight:850;letter-spacing:-.04em;margin:.25rem 0;}
.result-summary {font-size:1rem;line-height:1.65;color:#515762;}
.stat-card {background:var(--card);border:1px solid var(--line);border-radius:16px;padding:.85rem .9rem;min-height:100%;}
.stat-label {font-size:.76rem;color:var(--sub);font-weight:700;}
.stat-value {font-size:1.45rem;font-weight:850;margin-top:.15rem;}
.stat-note {font-size:.78rem;color:var(--sub);margin-top:.1rem;}
.strength-card {background:linear-gradient(135deg,#f4faf6,#ffffff);border:1px solid #dcebe2;border-radius:18px;padding:1rem;margin:.65rem 0;}
.strength-title {font-size:.84rem;font-weight:800;color:#44735d;margin-bottom:.25rem;}
.strength-copy {font-size:.92rem;color:#4c5c53;line-height:1.6;}
.bar-wrap {margin:.6rem 0 .8rem;}
.bar-top {display:flex;justify-content:space-between;gap:1rem;font-size:.84rem;color:#555b65;margin-bottom:.28rem;}
.bar-top strong {color:#2b3038;}
.bar-track {height:10px;border-radius:999px;background:#eeeae3;overflow:hidden;}
.bar-fill {height:100%;border-radius:999px;}
.bar-positive {background:#6f9c86;}
.bar-negative {background:#ba7477;}
.bar-gold {background:#b99a6d;}
.signal-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:.55rem;margin-top:.6rem;}
.signal-card {border:1px solid var(--line);border-radius:14px;padding:.7rem .75rem;background:#fcfbf8;}
.signal-label {font-size:.75rem;color:var(--sub);}
.signal-value {font-size:1.15rem;font-weight:850;margin-top:.05rem;}
.action-box {background:linear-gradient(135deg,#f4efe7,#faf8f4);border:1px solid #e4d6c2;border-radius:18px;padding:1rem 1.05rem;margin-top:.8rem;}
.action-title {font-size:.85rem;color:#886a42;font-weight:800;margin-bottom:.45rem;}
.soft-note {font-size:.8rem;color:var(--sub);line-height:1.5;}
[data-testid="stMetric"] {background:var(--card);border:1px solid var(--line);padding:.7rem .8rem;border-radius:15px;}
[data-testid="stMetricValue"] {font-size:1.35rem;font-weight:850;}
.stButton>button,[data-testid="stFormSubmitButton"]>button {border-radius:999px;min-height:2.75rem;font-weight:700;}
.stTextInput input {border-radius:13px!important;}
[data-testid="stExpander"] {border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.7);}
@media(max-width:700px){.hero-title{font-size:2rem}.signal-grid{grid-template-columns:1fr}.block-container{padding-left:1rem;padding-right:1rem}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-eyebrow">DIKE'S EYE · BALANCED DECISION</div>
  <div class="hero-title">⚖️ 좋은 점과 위험을 함께 보고 판단합니다</div>
  <div class="hero-sub">긍정 후기가 반복되는 강점은 제대로 반영하고, 의견이 갈리는 이유와 내 조건에서 실제로 문제가 되는 위험만 따로 봅니다. Evidence는 결론을 설명하는 근거이지, 결론 그 자체가 아닙니다.</div>
  <div class="hero-chips"><span class="hero-chip">Strength · 반복 강점</span><span class="hero-chip">Rashomon · 찬반 비교</span><span class="hero-chip">Wald · 놓친 신호</span></div>
</div>
""",
    unsafe_allow_html=True,
)

DEFAULTS = {"intent":None,"candidates":[],"search_status":"","selected_target":None,"analysis":None,"user_report":None,"explanation":None,"last_error":""}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.container(border=True):
    st.markdown('<div class="section-title">무엇을 고민하고 있나요?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">대상과 상황을 한 문장으로 적어주세요. 시간·목적·중요 조건까지 함께 쓰면 더 정확합니다.</div>', unsafe_allow_html=True)
    st.caption("예: 토요일 7시 소개팅인데 성수 어니언 어때?")
    st.caption("예: 출퇴근용으로 소니 WH-1000XM6 사도 될까? 배터리랑 착용감이 중요해")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("질문", placeholder="선택하려는 대상과 상황을 입력하세요", label_visibility="collapsed")
        submitted = st.form_submit_button("Dike에게 물어보기", type="primary", use_container_width=True)

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
    st.markdown('<div class="section-title">제가 이렇게 이해했어요</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns([2.5,1])
        with c1:
            st.markdown(f"**{kind_label} · {intent.get('target','')}**")
            bits = [intent.get("date_or_day",""),intent.get("time",""),intent.get("purpose",""),intent.get("preference","")]
            st.caption(" · ".join(x for x in bits if x) or "추가 조건 없음")
        with c2:
            st.metric("질문 해석", f"{int(intent.get('parse_confidence',0)*100)}%")

    if kind == "restaurant" and not st.session_state.selected_target:
        if st.button("NAVER에서 장소 확인", type="primary", use_container_width=True):
            with st.spinner("장소를 확인하고 있어요..."):
                candidates, status = local_search(intent.get("target") or intent.get("original", ""), **naver_credentials())
            st.session_state.search_status = status
            st.session_state.candidates = candidates or [{"title":intent.get("target") or intent.get("original", ""),"category":"직접 입력","address":"","fallback":True}]
        if st.session_state.candidates:
            labels = [f"{x.get('title','')} · {x.get('category','')} · {x.get('address','')}" for x in st.session_state.candidates]
            idx = st.radio("이 장소가 맞나요?", range(len(labels)), format_func=lambda i: labels[i])
            chosen = st.session_state.candidates[idx]
            if st.button("이 장소로 분석하기", type="primary", use_container_width=True):
                st.session_state.selected_target = {"kind":"restaurant","name":chosen.get("title") or intent.get("target", ""),"meta":chosen}

    if kind == "product" and not st.session_state.selected_target:
        with st.form("product_confirm_form"):
            product_name = st.text_input("제품명 / 모델명", key="target_edit")
            product_confirmed = st.form_submit_button("이 제품으로 분석하기", type="primary", use_container_width=True)
        if product_confirmed and product_name.strip():
            st.session_state.selected_target = {"kind":"product","name":product_name.strip(),"meta":{}}

selected = st.session_state.selected_target
if selected and not st.session_state.analysis:
    st.divider()
    st.markdown('<div class="section-title">내 조건을 한 번만 확인할게요</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">같은 리뷰라도 내 목적과 상황에 맞는 Evidence에 더 높은 가중치를 줍니다.</div>', unsafe_allow_html=True)
    with st.form("context_form"):
        if selected["kind"] == "restaurant":
            a,b = st.columns(2)
            with a:
                st.text_input("방문 날짜/요일", key="ctx_day", placeholder="예: 토요일")
                st.text_input("시간", key="ctx_time", placeholder="예: 19:00")
            with b:
                st.text_input("목적", key="ctx_purpose", placeholder="예: 소개팅")
                st.text_input("중요 조건", key="ctx_preference", placeholder="예: 조용함, 웨이팅")
        else:
            a,b = st.columns(2)
            with a:
                st.text_input("사용 목적", key="ctx_purpose", placeholder="예: 출퇴근, 업무")
                st.text_input("중요 조건", key="ctx_preference", placeholder="예: 배터리, 착용감")
            with b:
                st.text_input("사용 상황", key="ctx_day", placeholder="예: 매일")
                st.text_input("추가 조건", key="ctx_time", placeholder="예: 하루 3시간")
        analyze = st.form_submit_button("내 조건으로 판단하기", type="primary", use_container_width=True)
    if analyze:
        context = {"date_or_day":st.session_state.ctx_day,"time":st.session_state.ctx_time,"purpose":st.session_state.ctx_purpose,"preference":st.session_state.ctx_preference}
        try:
            with st.spinner("장점과 위험을 함께 비교하고 있어요..."):
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
    comps = d.get("components", {})
    total_count = len(a.get("rows", [])) + len(a.get("hidden_rows", []))
    visible_count = len(a.get("rows", []))
    hidden_count = len(a.get("hidden_rows", []))

    st.divider()
    st.markdown(
        f"""
        <div class="result-hero">
          <div class="result-label">Dike's View</div>
          <div class="result-title">{r.get('headline','')}</div>
          <div class="result-summary">{r.get('summary','')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s1,s2,s3 = st.columns(3)
    with s1:
        st.markdown(f"<div class='stat-card'><div class='stat-label'>조건 적합도</div><div class='stat-value'>{d.get('fit_score',0):.0f}/100</div><div class='stat-note'>장점과 위험 종합</div></div>", unsafe_allow_html=True)
    with s2:
        st.markdown(f"<div class='stat-card'><div class='stat-label'>판단 신뢰도</div><div class='stat-value'>{d.get('confidence',0):.0f}%</div><div class='stat-note'>Evidence 품질 반영</div></div>", unsafe_allow_html=True)
    with s3:
        st.markdown(f"<div class='stat-card'><div class='stat-label'>검토 Evidence</div><div class='stat-value'>{total_count}건</div><div class='stat-note'>리뷰 {visible_count} + 이탈 {hidden_count}</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.3rem">먼저, 좋았던 점부터 볼게요</div>', unsafe_allow_html=True)
    positive_aspects = comps.get("positive_aspects", [])
    if positive_aspects:
        for item in positive_aspects[:3]:
            label = ASPECT_LABELS.get(str(item.get("aspect")), str(item.get("aspect")))
            pos = int(item.get("positive_count", 0))
            neg = int(item.get("negative_count", 0))
            rate = float(item.get("positive_rate", 0.0)) * 100
            with st.container(border=True):
                st.markdown(f"**✨ {label}**")
                bar_row("긍정", rate, f"{pos}건 · {rate:.0f}%", "positive")
                st.caption(f"이 항목은 긍정 {pos}건 / 부정 {neg}건으로, 반복적으로 확인되는 강점으로 반영했어요.")
    else:
        st.info("현재 Evidence에서는 반복적으로 확인되는 뚜렷한 긍정 강점이 아직 충분하지 않습니다.")

    conflicts = a.get("rca", {}).get("conflicts", [])
    if conflicts:
        p = conflicts[0]
        label = ASPECT_LABELS.get(str(p.get("aspect")), str(p.get("aspect")))
        pos_count = int(p.get("positive_count", 0))
        neg_count = int(p.get("negative_count", 0))
        pos_rate = float(p.get("positive_rate", 0.0)) * 100
        neg_rate = float(p.get("negative_rate", 0.0)) * 100
        with st.container(border=True):
            st.markdown(f"**🎭 {label} — 의견이 갈린 지점**")
            bar_row("긍정", pos_rate, f"{pos_count}건 · {pos_rate:.0f}%", "positive")
            bar_row("부정", neg_rate, f"{neg_count}건 · {neg_rate:.0f}%", "negative")
            st.caption("이 항목은 좋다는 의견과 아쉽다는 의견이 함께 있어, 평균값만으로 판단하기 어려운 부분이에요.")

    rca_candidates = a.get("rca", {}).get("cause_candidates", [])
    aligned = [x for x in rca_candidates if x.get("user_aligned")]
    rca_point = (aligned or rca_candidates or [None])[0]
    if rca_point:
        label = ASPECT_LABELS.get(str(rca_point.get("aspect")), str(rca_point.get("aspect")))
        ctx = str(rca_point.get("context") or "특정 조건")
        base_total = int(rca_point.get("baseline_total_count", 0))
        base_neg = int(rca_point.get("baseline_negative_count", 0))
        ctx_total = int(rca_point.get("context_total_count", 0))
        ctx_neg = int(rca_point.get("context_negative_count", 0))
        base_rate = float(rca_point.get("baseline_negative_rate", 0.0)) * 100
        ctx_rate = float(rca_point.get("context_negative_rate", 0.0)) * 100
        diff = float(rca_point.get("lift", 0.0)) * 100
        with st.container(border=True):
            st.markdown(f"**🔎 {ctx} · {label} — 내 조건에서 달라진 부분**")
            c1,c2,c3 = st.columns(3)
            c1.metric("전체 부정", f"{base_rate:.0f}%", f"{base_neg}/{base_total}건")
            c2.metric("해당 조건 부정", f"{ctx_rate:.0f}%", f"{ctx_neg}/{ctx_total}건")
            c3.metric("차이", f"{diff:+.1f}%p")
            bar_row("전체", base_rate, f"{base_rate:.0f}%", "gold")
            bar_row("내 조건", ctx_rate, f"{ctx_rate:.0f}%", "negative" if diff > 0 else "positive")
            if diff > 0:
                st.caption("내 조건에서는 부정 의견이 전체보다 더 자주 나타났어요. 이 부분만 실제 선택에서 주의하면 됩니다.")
            else:
                st.caption("내 조건에서는 오히려 부정 의견이 전체보다 적었어요. 이 조건은 긍정적으로 반영할 수 있습니다.")

    signal_counts = a.get("wald", {}).get("signal_counts", {})
    if signal_counts:
        with st.container(border=True):
            st.markdown("**🕳️ 리뷰만 보면 놓칠 수 있는 신호**")
            top_signals = sorted(signal_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            cards = "".join([
                f"<div class='signal-card'><div class='signal-label'>{WALD_LABELS.get(k,k)}</div><div class='signal-value'>{v}건</div></div>"
                for k,v in top_signals
            ])
            st.markdown(f"<div class='signal-grid'>{cards}</div>", unsafe_allow_html=True)
            st.caption("이 숫자는 실제 발생률이 아니라 검색된 이탈·실패 신호의 건수예요. 그래서 위험 보정은 제한적으로만 적용합니다.")

    if r.get("strengths") or r.get("findings"):
        with st.container(border=True):
            st.markdown("**📌 분석 결과를 한 번에 정리하면**")
            for item in (r.get("strengths", [])[:2] + r.get("findings", [])[:2]):
                st.markdown(f"- {item}")

    if r.get("conflicting_evidence") or r.get("limitations"):
        with st.expander("같이 봐야 하는 주의점"):
            for item in (r.get("conflicting_evidence", []) + r.get("limitations", [])):
                st.markdown(f"- {item}")

    st.markdown('<div class="action-box"><div class="action-title">⚖️ Dike의 최종 제안</div>', unsafe_allow_html=True)
    for item in r.get("recommendations", []):
        st.markdown(f"- **{item}**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(r.get("closing_note", ""))

    with st.expander("AI가 이 결과를 조금 더 자연스럽게 설명하기"):
        st.caption("점수와 추천 여부는 이미 확정되어 있고, AI는 숫자와 근거를 읽기 쉽게 설명만 합니다.")
        if st.button("설명 생성", use_container_width=True):
            with st.spinner("결과를 정리하고 있어요..."):
                st.session_state.explanation = generate_explanation(a, api_key=secret("OPENAI_API_KEY"), model=secret("OPENAI_MODEL","gpt-5-mini"))
        if st.session_state.explanation:
            e = st.session_state.explanation
            st.markdown(f"**{e.get('headline','')}**")
            st.write(e.get("answer",""))
            for reason in e.get("reasons",[]):
                st.markdown(f"- {reason}")

    with st.expander("분석 근거 자세히 보기"):
        st.markdown("##### 점수 구성")
        st.json({
            "weighted_sentiment": comps.get("weighted_sentiment"),
            "context_sentiment": comps.get("context_sentiment"),
            "positive_strength": comps.get("positive_strength"),
            "rca_risk": comps.get("rca_risk"),
            "wald_risk": comps.get("wald_risk"),
            "evidence_quality": comps.get("evidence_quality"),
            "policy": d.get("policy", {}),
        })
        st.markdown("##### RCA · 조건별 차이")
        rca_df = pd.DataFrame(rca_candidates)
        if not rca_df.empty:
            cols = [c for c in ["aspect","context","baseline_total_count","baseline_negative_count","baseline_negative_rate","context_total_count","context_negative_count","context_negative_rate","lift","confidence","user_aligned"] if c in rca_df.columns]
            st.dataframe(rca_df[cols], use_container_width=True, hide_index=True)
        else:
            st.info("조건별 차이를 설명할 만큼 충분한 후보가 없습니다.")
        st.markdown("##### 상위 Evidence")
        df = pd.DataFrame(a.get("rows", []))
        cols = [c for c in ["source","title","aspects","contexts","sentiment","R","F","M","priority"] if c in df.columns]
        if not df.empty:
            st.dataframe(df[cols].head(20), use_container_width=True, hide_index=True)

    if st.button("새로운 질문하기", use_container_width=True):
        for key in list(DEFAULTS) + ["ctx_day","ctx_time","ctx_purpose","ctx_preference","target_edit"]:
            st.session_state.pop(key, None)
        st.rerun()

if st.session_state.last_error:
    st.error("처리 중 문제가 생겼어요. 입력 내용은 유지되어 있습니다.")
    with st.expander("오류 정보"):
        st.code(st.session_state.last_error)

st.divider()
st.caption("Dike's Eye · 장점과 위험을 함께 보고 내 조건에서의 선택을 돕는 Decision Agent")

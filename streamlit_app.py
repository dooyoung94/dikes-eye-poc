import html
import os

import pandas as pd
import streamlit as st

from src.condition_analysis import normalize_context_conditions
from src.condition_taxonomy import aspect_label
from src.eda import build_eda
from src.intent import parse_intent
from src.llm_explain import generate_explanation
from src.naver_client import collect_hidden_evidence, collect_visible_evidence, local_search
from src.normalize import normalize_evidence
from src.rca import derive_rca
from src.rashomon import build_rashomon
from src.reporting import WALD_LABELS, build_user_report
from src.rfm import build_rfm
from src.scoring import score_decision
from src.wald import analyze_wald

st.set_page_config(
    page_title="Dike's Eye · Conditional Decision Agent",
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


def intent_context(intent: dict | None) -> dict:
    intent = intent or {}
    return {
        "date_or_day": str(intent.get("date_or_day") or "").strip(),
        "time": str(intent.get("time") or "").strip(),
        "purpose": str(intent.get("purpose") or "").strip(),
        "preference": str(intent.get("preference") or "").strip(),
        "conditions": intent.get("conditions", []) if isinstance(intent.get("conditions"), list) else [],
    }


def reset_after_question(intent: dict) -> None:
    parsed_context = intent_context(intent)
    st.session_state.intent = intent
    st.session_state.parsed_context = parsed_context
    st.session_state.candidates = []
    st.session_state.search_status = ""
    st.session_state.selected_target = None
    st.session_state.analysis = None
    st.session_state.user_report = None
    st.session_state.explanation = None
    st.session_state.last_error = ""
    st.session_state.ctx_day = parsed_context["date_or_day"]
    st.session_state.ctx_time = parsed_context["time"]
    st.session_state.ctx_purpose = parsed_context["purpose"]
    st.session_state.ctx_preference = parsed_context["preference"]
    st.session_state.target_edit = str(intent.get("target") or "")


def current_context() -> dict:
    parsed = st.session_state.get("parsed_context") or intent_context(st.session_state.get("intent"))
    values = {
        "date_or_day": str(st.session_state.get("ctx_day") or "").strip(),
        "time": str(st.session_state.get("ctx_time") or "").strip(),
        "purpose": str(st.session_state.get("ctx_purpose") or "").strip(),
        "preference": str(st.session_state.get("ctx_preference") or "").strip(),
    }
    for key in values:
        if not values[key]:
            values[key] = str(parsed.get(key) or "").strip()

    if values["preference"] == str(parsed.get("preference") or "").strip():
        values["conditions"] = parsed.get("conditions", [])
    else:
        values["conditions"] = normalize_context_conditions({"preference": values["preference"]})
    return values


def finalize_context(target: str) -> dict:
    context = current_context()
    parsed = st.session_state.get("parsed_context") or {}
    preference_changed = (
        str(context.get("preference") or "").strip()
        != str(parsed.get("preference") or "").strip()
    )
    if preference_changed and context.get("preference"):
        try:
            reparsed = parse_intent(
                f"{target} 어때? {context['preference']}",
                api_key=secret("OPENAI_API_KEY"),
                model=secret("OPENAI_MODEL", "gpt-5-mini"),
            )
            if reparsed.get("conditions"):
                context["conditions"] = reparsed["conditions"]
        except Exception:
            pass
    return context


def direction_label(direction: str) -> str:
    return {
        "prefer": "중요하게 봄",
        "avoid": "피하고 싶음",
        "tolerate": "어느 정도 허용",
    }.get(str(direction), "중요하게 봄")


def context_items(context: dict, kind: str) -> list[tuple[str, str]]:
    base = [
        ("요일/상황", context.get("date_or_day", "")),
        ("시간", context.get("time", "")),
        ("목적", context.get("purpose", "")),
    ]
    return [(label, str(value).strip()) for label, value in base if str(value).strip()]


def render_context_chips(context: dict, kind: str, title: str = "이번 판단 조건") -> None:
    items = context_items(context, kind)
    conditions = normalize_context_conditions(context)
    chips = []
    for label, value in items:
        chips.append(
            f"<span class='context-chip'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></span>"
        )
    for cond in conditions:
        chips.append(
            "<span class='context-chip condition-chip'>"
            f"<span>{html.escape(direction_label(str(cond.get('direction'))))}</span>"
            f"<strong>{html.escape(str(cond.get('raw') or cond.get('label') or ''))}</strong>"
            "</span>"
        )
    body = "".join(chips) if chips else "<span class='context-empty'>별도 조건이 없습니다.</span>"
    st.markdown(
        f"<div class='context-panel'><div class='context-title'>🎯 {html.escape(title)}</div><div class='context-chips'>{body}</div></div>",
        unsafe_allow_html=True,
    )


def run_decision(target: str, kind: str, context: dict) -> dict:
    creds = naver_credentials()
    visible_raw = collect_visible_evidence(target, kind=kind, context=context, **creds)
    hidden_raw = collect_hidden_evidence(target, kind=kind, context=context, **creds)
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
        "context": dict(context),
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
        f"<div class='bar-wrap'><div class='bar-top'><span>{html.escape(label)}</span><strong>{html.escape(count_text)}</strong></div>"
        f"<div class='bar-track'><div class='bar-fill {cls}' style='width:{width:.1f}%'></div></div></div>",
        unsafe_allow_html=True,
    )


st.markdown(
    """
<style>
:root{--bg:#f7f5f0;--card:#fff;--ink:#20242d;--sub:#70747d;--gold:#aa8351;--line:#ebe6dd;--green:#5f927a;--red:#b8686c}
html,body,[class*="css"]{font-family:Pretendard,-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Segoe UI",sans-serif}
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#fbfaf7 0%,var(--bg) 100%);color:var(--ink)}
[data-testid="stHeader"]{background:transparent}.block-container{max-width:920px;padding-top:1.3rem;padding-bottom:5rem}
.hero{position:relative;overflow:hidden;background:linear-gradient(135deg,#202631,#303948);color:white;border-radius:28px;padding:1.65rem 1.7rem;margin-bottom:1rem;box-shadow:0 18px 45px rgba(25,29,36,.12)}
.hero:after{content:"⚖";position:absolute;right:1rem;top:-1.5rem;font-size:9rem;opacity:.075}.hero-eyebrow{font-size:.72rem;font-weight:800;letter-spacing:.15em;color:#dbc29b}.hero-title{font-size:2.25rem;font-weight:850;letter-spacing:-.045em;margin:.3rem 0 .45rem}.hero-sub{font-size:1rem;line-height:1.65;color:#e6e9ee;max-width:720px}.hero-chips{display:flex;gap:.45rem;flex-wrap:wrap;margin-top:.9rem}.hero-chip{font-size:.76rem;padding:.34rem .68rem;border:1px solid rgba(255,255,255,.18);border-radius:999px;background:rgba(255,255,255,.06)}
.section-title{font-size:1.28rem;font-weight:820;letter-spacing:-.025em;margin:.25rem 0 .25rem}.section-sub{font-size:.9rem;color:var(--sub);margin-bottom:.8rem}.result-hero{background:var(--card);border:1px solid var(--line);border-radius:22px;padding:1.15rem 1.2rem;box-shadow:0 10px 30px rgba(31,35,41,.05)}.result-label{font-size:.72rem;font-weight:800;color:var(--gold);letter-spacing:.12em}.result-title{font-size:1.75rem;font-weight:850;letter-spacing:-.035em;margin:.25rem 0}.result-summary{font-size:.98rem;line-height:1.62;color:#535964}
.context-panel{background:linear-gradient(135deg,#fffdf8,#f7f2e9);border:1px solid #e7dbc9;border-radius:18px;padding:.85rem .95rem;margin:.7rem 0}.context-title{font-size:.81rem;font-weight:850;color:#856740;margin-bottom:.5rem}.context-chips{display:flex;gap:.42rem;flex-wrap:wrap}.context-chip{display:inline-flex;align-items:center;gap:.32rem;background:#fff;border:1px solid #e7dfd2;border-radius:999px;padding:.38rem .65rem;font-size:.78rem}.context-chip span{color:#858993}.context-chip strong{color:#252a31}.condition-chip{border-color:#d8c39f;background:#fffaf1}
.stat-card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:.82rem .86rem}.stat-label{font-size:.74rem;color:var(--sub);font-weight:700}.stat-value{font-size:1.4rem;font-weight:850;margin-top:.1rem}.stat-note{font-size:.76rem;color:var(--sub)}
.condition-card{background:#fff;border:1px solid var(--line);border-radius:20px;padding:1rem 1.05rem;margin:.7rem 0;box-shadow:0 6px 20px rgba(35,38,43,.035)}.condition-head{display:flex;justify-content:space-between;gap:.8rem;align-items:flex-start}.condition-name{font-size:1.06rem;font-weight:820}.condition-meta{font-size:.76rem;color:#8a6c43;background:#f7f0e5;border-radius:999px;padding:.3rem .55rem;white-space:nowrap}.condition-summary{font-size:.87rem;color:#626872;margin-top:.35rem;line-height:1.5}.bar-wrap{margin:.58rem 0 .7rem}.bar-top{display:flex;justify-content:space-between;gap:1rem;font-size:.82rem;color:#5b616a;margin-bottom:.25rem}.bar-top strong{color:#2c3138}.bar-track{height:9px;border-radius:999px;background:#eeeae3;overflow:hidden}.bar-fill{height:100%;border-radius:999px}.bar-positive{background:#6f9c86}.bar-negative{background:#ba7477}.bar-gold{background:#b99a6d}
.signal-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem}.signal-card{border:1px solid var(--line);border-radius:14px;padding:.68rem;background:#fcfbf8}.signal-label{font-size:.74rem;color:var(--sub)}.signal-value{font-size:1.12rem;font-weight:850}.action-box{background:linear-gradient(135deg,#f4efe7,#faf8f4);border:1px solid #e4d6c2;border-radius:18px;padding:1rem;margin-top:.8rem}.action-title{font-size:.84rem;color:#886a42;font-weight:800;margin-bottom:.42rem}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);padding:.65rem .75rem;border-radius:14px}[data-testid="stMetricValue"]{font-size:1.28rem;font-weight:850}.stButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:999px;min-height:2.7rem;font-weight:700}.stTextInput input{border-radius:13px!important}[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.72)}
@media(max-width:700px){.hero-title{font-size:1.85rem}.signal-grid{grid-template-columns:1fr}.block-container{padding-left:1rem;padding-right:1rem}.condition-head{display:block}.condition-meta{display:inline-block;margin-top:.35rem}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-eyebrow">DIKE'S EYE · CONDITIONAL DECISION AGENT</div>
  <div class="hero-title">⚖️ 평균이 아니라, 내 조건에서 선택해도 되는지 판단합니다</div>
  <div class="hero-sub">Dike는 사용자가 중요하게 말한 조건과 사용 상황을 먼저 해석합니다. 각 조건의 긍정·부정 Evidence를 따로 계산하고, 같은 대상도 시간·목적·상황이 달라지면 결론을 다시 조정합니다.</div>
  <div class="hero-chips"><span class="hero-chip">조건 해석 · LLM</span><span class="hero-chip">조건별 계산 · Deterministic</span><span class="hero-chip">Rashomon · 의견 충돌</span><span class="hero-chip">Wald · 사라진 신호</span></div>
</div>
""",
    unsafe_allow_html=True,
)

DEFAULTS = {"intent":None,"parsed_context":{},"candidates":[],"search_status":"","selected_target":None,"analysis":None,"user_report":None,"explanation":None,"last_error":""}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.container(border=True):
    st.markdown('<div class="section-title">어떤 선택을 고민하고 있나요?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">대상과 조건을 평소 말하듯 적어주세요. “비싸도 괜찮지만 아늑해야 해”, “웨이팅은 싫어” 같은 뉘앙스도 조건으로 해석합니다.</div>', unsafe_allow_html=True)
    st.caption("예: 야키니쿠 하코 어때? 가격은 좀 있어도 괜찮고, 안락함과 분위기가 중요해")
    st.caption("예: 토요일 7시 소개팅인데 성수 어니언 어때? 조용해야 하고 웨이팅은 싫어")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("질문", placeholder="대상 + 내 상황 + 중요조건을 입력하세요", label_visibility="collapsed")
        submitted = st.form_submit_button("Dike에게 판단 맡기기", type="primary", use_container_width=True)

if submitted and question.strip():
    try:
        with st.spinner("질문의 조건과 허용 범위를 읽고 있어요..."):
            parsed = parse_intent(
                question.strip(),
                api_key=secret("OPENAI_API_KEY"),
                model=secret("OPENAI_MODEL", "gpt-5-mini"),
            )
        reset_after_question(parsed)
    except Exception as exc:
        st.session_state.last_error = f"{type(exc).__name__}: {exc}"

intent = st.session_state.intent
if intent:
    kind = intent.get("kind", "restaurant")
    kind_label = "식당" if kind == "restaurant" else "상품"
    st.divider()
    st.markdown('<div class="section-title">Dike가 질문을 이렇게 이해했어요</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns([2.5,1])
        with c1:
            st.markdown(f"**{kind_label} · {html.escape(str(intent.get('target','')))}**")
            st.caption(f"조건 해석 방식 · {'LLM' if intent.get('parser_source') == 'openai' else '규칙 기반 fallback'}")
        with c2:
            st.metric("해석 신뢰도", f"{int(float(intent.get('parse_confidence',0))*100)}%")
        render_context_chips(current_context(), kind, "질문에서 읽은 조건")

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
            if st.button("이 장소가 맞아요", type="primary", use_container_width=True):
                st.session_state.selected_target = {"kind":"restaurant","name":chosen.get("title") or intent.get("target", ""),"meta":chosen}

    if kind == "product" and not st.session_state.selected_target:
        with st.form("product_confirm_form"):
            product_name = st.text_input("제품명 / 모델명", key="target_edit")
            confirmed = st.form_submit_button("이 제품이 맞아요", type="primary", use_container_width=True)
        if confirmed and product_name.strip():
            st.session_state.selected_target = {"kind":"product","name":product_name.strip(),"meta":{}}

selected = st.session_state.selected_target
if selected and not st.session_state.analysis:
    st.divider()
    st.markdown('<div class="section-title">이 조건을 기준으로 판단할게요</div>', unsafe_allow_html=True)
    render_context_chips(current_context(), selected["kind"], "분석 기준")
    with st.expander("조건이 다르면 수정"):
        a,b = st.columns(2)
        with a:
            st.text_input("요일/사용 상황", key="ctx_day")
            st.text_input("시간", key="ctx_time")
        with b:
            st.text_input("목적", key="ctx_purpose")
            st.text_input("중요 조건", key="ctx_preference", help="예: 가격은 좀 비싸도 괜찮고, 분위기와 안락함이 중요해")
        st.caption("중요 조건을 수정하면 분석 직전에 LLM이 조건의 방향·중요도를 다시 해석합니다.")
    if st.button("이 조건으로 판단하기", type="primary", use_container_width=True):
        context = finalize_context(selected["name"])
        try:
            with st.spinner("조건별 Evidence와 상황별 변화를 계산하고 있어요..."):
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
    rows = a.get("rows", [])
    hidden_rows = a.get("hidden_rows", [])
    result_context = a.get("context", {})
    condition_results = a.get("rca", {}).get("condition_results", [])

    st.divider()
    st.markdown(
        f"<div class='result-hero'><div class='result-label'>DIKE'S CONDITIONAL VIEW</div>"
        f"<div class='result-title'>{html.escape(str(r.get('headline','')))}</div>"
        f"<div class='result-summary'>{html.escape(str(r.get('summary','')))}</div></div>",
        unsafe_allow_html=True,
    )
    render_context_chips(result_context, a.get("kind", "restaurant"), "이번 판단에 사용한 조건")

    s1,s2,s3 = st.columns(3)
    s1.markdown(f"<div class='stat-card'><div class='stat-label'>조건 적합도</div><div class='stat-value'>{d.get('fit_score',0):.0f}/100</div><div class='stat-note'>조건 중요도까지 반영</div></div>", unsafe_allow_html=True)
    s2.markdown(f"<div class='stat-card'><div class='stat-label'>판단 신뢰도</div><div class='stat-value'>{d.get('confidence',0):.0f}%</div><div class='stat-note'>Evidence 품질·범위</div></div>", unsafe_allow_html=True)
    s3.markdown(f"<div class='stat-card'><div class='stat-label'>검토 Evidence</div><div class='stat-value'>{len(rows)+len(hidden_rows)}건</div><div class='stat-note'>Visible {len(rows)} + Hidden {len(hidden_rows)}</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.25rem">내가 중요하게 본 조건별 판단</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">전용 검색 결과만 보지 않고, 각 조건과 연결되는 전체 Evidence를 먼저 계산합니다. 전용 검색은 근거 범위를 보강하는 용도로만 사용합니다.</div>', unsafe_allow_html=True)

    if condition_results:
        for item in condition_results:
            label = str(item.get("label") or aspect_label(str(item.get("aspect"))))
            total = int(item.get("total_count",0))
            pos = int(item.get("positive_count",0))
            neg = int(item.get("negative_count",0))
            pos_rate = float(item.get("positive_rate",0))*100
            neg_rate = float(item.get("negative_rate",0))*100
            direct = int(item.get("direct_count",0))
            fit = float(item.get("fit",0.5))*100
            importance = float(item.get("importance",0.8))*100
            direction = direction_label(str(item.get("direction")))
            with st.container(border=False):
                st.markdown(
                    f"<div class='condition-card'><div class='condition-head'><div><div class='condition-name'>🎯 {html.escape(label)}</div>"
                    f"<div class='condition-summary'>{html.escape(str(item.get('raw') or label))} · 전체 관련 Evidence {total}건 · 직접 검색/표현 근거 {direct}건</div></div>"
                    f"<div class='condition-meta'>{html.escape(direction)} · 중요도 {importance:.0f}%</div></div></div>",
                    unsafe_allow_html=True,
                )
                if total >= 3:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("긍정", f"{pos_rate:.0f}%", f"{pos}/{total}건")
                    c2.metric("부정", f"{neg_rate:.0f}%", f"{neg}/{total}건")
                    c3.metric("조건 적합", f"{fit:.0f}/100")
                    bar_row("긍정 Evidence", pos_rate, f"{pos}건 · {pos_rate:.0f}%", "positive")
                    if int(item.get("situational_count",0)) >= 3:
                        situ_rate = float(item.get("situational_negative_rate",0))*100
                        lift = float(item.get("situational_lift",0))*100
                        st.caption(
                            f"내 사용 상황에 직접 맞는 {int(item.get('situational_count',0))}건에서는 부정 {situ_rate:.0f}%로, 전체 대비 {lift:+.1f}%p 차이가 났습니다."
                        )
                    else:
                        st.caption("이 조건 자체는 평가할 수 있지만, 요일·시간·목적에 따른 추가 변화는 비교할 Evidence가 아직 충분하지 않습니다.")
                else:
                    st.warning(f"{label} 관련 의견 Evidence가 {total}건이라 조건 자체를 안정적으로 평가하기에는 부족합니다.")
    else:
        st.info("질문에서 별도의 중요조건을 찾지 못해 전체 Evidence 중심으로 판단했습니다.")

    conflicts = a.get("rca", {}).get("conflicts", [])
    if conflicts:
        st.markdown('<div class="section-title" style="margin-top:1.2rem">의견이 특히 갈린 부분</div>', unsafe_allow_html=True)
        for p in conflicts[:2]:
            label = aspect_label(str(p.get("aspect")))
            pos_count = int(p.get("positive_count",0)); neg_count = int(p.get("negative_count",0))
            pos_rate = float(p.get("positive_rate",0))*100; neg_rate = float(p.get("negative_rate",0))*100
            with st.container(border=True):
                st.markdown(f"**🎭 {label}** · 의견 {pos_count+neg_count}건")
                bar_row("긍정", pos_rate, f"{pos_count}건 · {pos_rate:.0f}%", "positive")
                bar_row("부정", neg_rate, f"{neg_count}건 · {neg_rate:.0f}%", "negative")

    signal_counts = a.get("wald", {}).get("signal_counts", {})
    if signal_counts:
        with st.container(border=True):
            st.markdown("**🕳️ 리뷰만 보면 놓칠 수 있는 신호**")
            top = sorted(signal_counts.items(), key=lambda x:x[1], reverse=True)[:3]
            cards = "".join(f"<div class='signal-card'><div class='signal-label'>{html.escape(WALD_LABELS.get(k,k))}</div><div class='signal-value'>{v}건</div></div>" for k,v in top)
            st.markdown(f"<div class='signal-grid'>{cards}</div>", unsafe_allow_html=True)
            st.caption("이 수치는 실제 발생률이 아니라 검색된 이탈·실패 신호 건수입니다.")

    positive_aspects = comps.get("positive_aspects", [])
    if positive_aspects:
        with st.expander("조건과 별개로 반복적으로 확인된 전체 강점"):
            for item in positive_aspects[:3]:
                label = aspect_label(str(item.get("aspect")))
                st.markdown(f"- **{label}** · 긍정 {item.get('positive_count',0)}건 / 부정 {item.get('negative_count',0)}건")

    st.markdown('<div class="action-box"><div class="action-title">⚖️ Dike의 최종 제안</div>', unsafe_allow_html=True)
    for item in r.get("recommendations", []):
        st.markdown(f"- **{item}**")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("AI가 이 판단을 자연스럽게 설명하기"):
        st.caption("점수·건수·비율은 계산 엔진이 확정합니다. AI는 결과를 설명만 합니다.")
        if st.button("설명 생성", use_container_width=True):
            with st.spinner("조건별 결과를 정리하고 있어요..."):
                st.session_state.explanation = generate_explanation(
                    a,
                    api_key=secret("OPENAI_API_KEY"),
                    model=secret("OPENAI_MODEL", "gpt-5-mini"),
                )
        if st.session_state.explanation:
            e = st.session_state.explanation
            st.markdown(f"**{e.get('headline','')}**")
            st.write(e.get("answer", ""))
            for reason in e.get("reasons", []):
                st.markdown(f"- {reason}")

    with st.expander("분석 근거 자세히 보기"):
        st.markdown("##### 조건별 계산")
        cdf = pd.DataFrame(condition_results)
        if not cdf.empty:
            cols = [c for c in ["label","raw","direction","importance","total_count","positive_count","negative_count","direct_count","situational_count","situational_lift","fit","evidence_confidence"] if c in cdf.columns]
            st.dataframe(cdf[cols], use_container_width=True, hide_index=True)
        st.markdown("##### 점수 구성")
        st.json({
            "weighted_sentiment": comps.get("weighted_sentiment"),
            "condition_fit": comps.get("condition_fit"),
            "condition_coverage": comps.get("condition_coverage"),
            "positive_strength": comps.get("positive_strength"),
            "rca_risk": comps.get("rca_risk"),
            "wald_risk": comps.get("wald_risk"),
            "policy": d.get("policy", {}),
        })
        st.markdown("##### 상위 Evidence")
        df = pd.DataFrame(rows)
        cols = [c for c in ["source","retrieval_scope","title","aspects","contexts","condition_direct_aspects","situational_aligned","sentiment","R","F","M","priority"] if c in df.columns]
        if not df.empty:
            st.dataframe(df[cols].head(40), use_container_width=True, hide_index=True)

    if st.button("새로운 질문하기", use_container_width=True):
        for key in list(DEFAULTS) + ["ctx_day","ctx_time","ctx_purpose","ctx_preference","target_edit"]:
            st.session_state.pop(key, None)
        st.rerun()

if st.session_state.last_error:
    st.error("처리 중 문제가 생겼어요. 입력한 내용은 유지되어 있습니다.")
    with st.expander("오류 정보"):
        st.code(st.session_state.last_error)

st.divider()
st.caption("Dike's Eye · Conditional Decision Agent · 자연어 조건은 LLM이 해석하고, Evidence·비율·점수는 deterministic engine이 계산합니다.")

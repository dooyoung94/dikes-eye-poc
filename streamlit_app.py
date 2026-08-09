import html
import os

import pandas as pd
import streamlit as st

from src.condition_analysis import normalize_context_conditions
from src.condition_taxonomy import aspect_label
from src.eda import build_eda
from src.evidence_insights import build_condition_evidence, build_conflict_insights, build_consensus
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


def parse_question(text: str) -> dict:
    try:
        return parse_intent(
            text,
            api_key=secret("OPENAI_API_KEY"),
            model=secret("OPENAI_MODEL", "gpt-5-mini"),
        )
    except TypeError:
        return parse_intent(text)


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
            reparsed = parse_question(f"{target} 어때? {context['preference']}")
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


def render_context_chips(context: dict, title: str = "이번 판단 조건") -> None:
    chips: list[str] = []
    for label, value in [
        ("요일/상황", context.get("date_or_day", "")),
        ("시간", context.get("time", "")),
        ("목적", context.get("purpose", "")),
    ]:
        if str(value).strip():
            chips.append(
                f"<span class='context-chip'><span>{html.escape(label)}</span>"
                f"<strong>{html.escape(str(value))}</strong></span>"
            )
    for cond in normalize_context_conditions(context):
        chips.append(
            "<span class='context-chip condition-chip'>"
            f"<span>{html.escape(direction_label(str(cond.get('direction'))))}</span>"
            f"<strong>{html.escape(str(cond.get('raw') or cond.get('label') or ''))}</strong>"
            "</span>"
        )
    body = "".join(chips) if chips else "<span class='context-empty'>별도 조건이 없습니다.</span>"
    st.markdown(
        f"<div class='context-panel'><div class='context-title'>🎯 {html.escape(title)}</div>"
        f"<div class='context-chips'>{body}</div></div>",
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
    consensus = build_consensus(rfm_rows)
    conflict_insights = build_conflict_insights(rfm_rows, rca.get("conflicts", []))
    condition_evidence = build_condition_evidence(rfm_rows, rca.get("condition_results", []))
    decision = score_decision(rfm_rows, eda, rfm_summary, rca, wald, consensus=consensus)
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
        "consensus": consensus,
        "conflict_insights": conflict_insights,
        "condition_evidence": condition_evidence,
        "decision": decision,
    }


def bar_row(label: str, value: float, count_text: str, tone: str = "gold") -> None:
    width = max(0, min(100, float(value)))
    cls = "bar-negative" if tone == "negative" else "bar-positive" if tone == "positive" else "bar-gold"
    st.markdown(
        f"<div class='bar-wrap'><div class='bar-top'><span>{html.escape(label)}</span>"
        f"<strong>{html.escape(count_text)}</strong></div>"
        f"<div class='bar-track'><div class='bar-fill {cls}' style='width:{width:.1f}%'></div></div></div>",
        unsafe_allow_html=True,
    )


def keyword_pills(items: list[dict], tone: str) -> str:
    cls = "kw-pos" if tone == "positive" else "kw-neg"
    if not items:
        return "<span class='muted'>뚜렷한 반복 키워드 없음</span>"
    return "".join(
        f"<span class='kw-pill {cls}'>{html.escape(str(x.get('keyword','')))} <b>{int(x.get('count',0))}</b></span>"
        for x in items[:5]
    )


def evidence_samples(title: str, samples: list[dict]) -> None:
    if not samples:
        return
    st.markdown(f"**{title}**")
    for sample in samples[:2]:
        heading = html.escape(str(sample.get("title") or sample.get("evidence_id") or "Evidence"))
        snippet = html.escape(str(sample.get("snippet") or ""))
        source = html.escape(str(sample.get("source") or ""))
        st.markdown(
            f"<div class='evidence-quote'><div class='evidence-source'>{source}</div>"
            f"<div class='evidence-title'>{heading}</div>"
            f"<div class='evidence-text'>{snippet}</div></div>",
            unsafe_allow_html=True,
        )


st.markdown(
    """
<style>
:root{--bg:#eee8da;--paper:#fffdf8;--ink:#24261f;--sub:#6f7168;--gold:#b18a4f;--bronze:#806238;--line:#ded4bf;--olive:#6f7650;--green:#628b71;--red:#ad6464}
html,body,[class*="css"]{font-family:Pretendard,-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Segoe UI",sans-serif}
[data-testid="stAppViewContainer"]{background:radial-gradient(circle at 12% 5%,rgba(174,153,109,.18),transparent 25rem),radial-gradient(circle at 92% 18%,rgba(99,106,73,.13),transparent 24rem),repeating-linear-gradient(90deg,rgba(128,111,77,.025) 0,rgba(128,111,77,.025) 1px,transparent 1px,transparent 70px),linear-gradient(180deg,#f7f2e8 0%,#eee7d9 48%,#f7f2e8 100%);color:var(--ink)}
[data-testid="stAppViewContainer"]:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.22;background-image:linear-gradient(115deg,transparent 0 47%,rgba(108,96,70,.05) 48%,transparent 49%),linear-gradient(70deg,transparent 0 72%,rgba(108,96,70,.04) 73%,transparent 74%);background-size:260px 210px,310px 250px}
[data-testid="stHeader"]{background:transparent}.block-container{max-width:940px;padding-top:1.25rem;padding-bottom:5rem}
.hero{position:relative;overflow:hidden;background:linear-gradient(90deg,rgba(30,35,32,.96),rgba(45,51,42,.93)),linear-gradient(135deg,#252b2a,#3a4236);color:white;border:1px solid rgba(208,184,132,.26);border-radius:30px;padding:2.15rem 2rem 2rem;margin-bottom:1rem;box-shadow:0 20px 55px rgba(54,49,37,.18)}
.hero:before,.hero:after{content:"";position:absolute;bottom:-28px;width:96px;height:230px;opacity:.10;border:1px solid #e8d7b4;border-bottom:none;border-radius:48px 48px 0 0}.hero:before{left:24px}.hero:after{right:24px}
.hero-center{position:relative;z-index:2;max-width:690px}.hero-eyebrow{font-size:.71rem;font-weight:800;letter-spacing:.19em;color:#d8bd89;margin-bottom:.65rem}.hero-title{font-size:2.4rem;font-weight:860;letter-spacing:-.05em;line-height:1.2;margin:.1rem 0 .65rem}.hero-title .gold{color:#ddc18c}.hero-sub{font-size:.98rem;line-height:1.75;color:#e3e5df;max-width:660px}.hero-seal{position:absolute;right:34px;top:20px;width:170px;height:170px;opacity:.31}.hero-chips{display:flex;gap:.45rem;flex-wrap:wrap;margin-top:1.05rem}.hero-chip{font-size:.75rem;padding:.35rem .7rem;border:1px solid rgba(224,204,165,.25);border-radius:999px;background:rgba(255,255,255,.055);color:#ece8de}
.section-title{font-size:1.28rem;font-weight:830;letter-spacing:-.025em;margin:.3rem 0 .25rem}.section-sub{font-size:.9rem;color:var(--sub);margin-bottom:.8rem;line-height:1.55}.result-hero,.consensus-card,.balance-card{background:rgba(255,253,248,.93);border:1px solid var(--line);border-radius:22px;padding:1.15rem 1.2rem;box-shadow:0 10px 28px rgba(71,60,38,.055)}
.result-label,.mini-eyebrow{font-size:.72rem;font-weight:850;color:var(--bronze);letter-spacing:.12em}.result-title{font-size:1.78rem;font-weight:860;letter-spacing:-.035em;margin:.28rem 0}.result-summary{font-size:.98rem;line-height:1.65;color:#565b54}
.context-panel{background:linear-gradient(135deg,#fffaf0,#f3ead8);border:1px solid #dccdac;border-radius:18px;padding:.85rem .95rem;margin:.7rem 0}.context-title{font-size:.81rem;font-weight:850;color:#77603c;margin-bottom:.5rem}.context-chips{display:flex;gap:.42rem;flex-wrap:wrap}.context-chip{display:inline-flex;align-items:center;gap:.32rem;background:#fffdf7;border:1px solid #dfd1b7;border-radius:999px;padding:.38rem .65rem;font-size:.78rem}.context-chip span{color:#85877f}.context-chip strong{color:#2a2c27}.condition-chip{border-color:#cabb92;background:#fff8e9}
.stat-card{background:rgba(255,253,248,.95);border:1px solid var(--line);border-radius:16px;padding:.82rem .86rem}.stat-label{font-size:.74rem;color:var(--sub);font-weight:700}.stat-value{font-size:1.4rem;font-weight:850;margin-top:.1rem}.stat-note{font-size:.76rem;color:var(--sub)}
.condition-card{background:#fffdf8;border:1px solid var(--line);border-radius:20px;padding:1rem 1.05rem;margin:.7rem 0;box-shadow:0 6px 18px rgba(54,47,35,.035)}.condition-head{display:flex;justify-content:space-between;gap:.8rem;align-items:flex-start}.condition-name{font-size:1.06rem;font-weight:830}.condition-meta{font-size:.76rem;color:#765f3e;background:#f2eadb;border-radius:999px;padding:.3rem .55rem;white-space:nowrap}.condition-summary{font-size:.86rem;color:#676b64;margin-top:.35rem;line-height:1.5}
.bar-wrap{margin:.58rem 0 .7rem}.bar-top{display:flex;justify-content:space-between;gap:1rem;font-size:.82rem;color:#5f625d;margin-bottom:.25rem}.bar-top strong{color:#2c2f2a}.bar-track{height:9px;border-radius:999px;background:#ebe5da;overflow:hidden}.bar-fill{height:100%;border-radius:999px}.bar-positive{background:#708c72}.bar-negative{background:#ae6c69}.bar-gold{background:#b89a66}
.kw-row{display:flex;flex-wrap:wrap;gap:.38rem;margin:.45rem 0 .75rem}.kw-pill{display:inline-flex;gap:.28rem;align-items:center;border-radius:999px;padding:.32rem .58rem;font-size:.75rem;border:1px solid}.kw-pos{background:#f0f5ef;border-color:#c9dac9;color:#496451}.kw-neg{background:#f8eeee;border-color:#e5c9c9;color:#814e4e}.muted{color:#8c8f88;font-size:.78rem}
.evidence-quote{border-left:3px solid #b7a072;background:#faf7ef;border-radius:0 12px 12px 0;padding:.68rem .75rem;margin:.45rem 0}.evidence-source{font-size:.68rem;color:#8b806a;font-weight:800;text-transform:uppercase}.evidence-title{font-size:.82rem;font-weight:800;margin:.12rem 0}.evidence-text{font-size:.8rem;line-height:1.52;color:#62645f}
.signal-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.5rem}.signal-card{border:1px solid var(--line);border-radius:14px;padding:.68rem;background:#fcf9f2}.signal-label{font-size:.74rem;color:var(--sub)}.signal-value{font-size:1.12rem;font-weight:850}
.balance-grid{display:grid;grid-template-columns:1fr 70px 1fr;align-items:center;gap:.8rem;margin-top:.8rem}.balance-side{text-align:center;background:#faf6ed;border:1px solid #e2d7c2;border-radius:16px;padding:.85rem}.balance-score{font-size:1.65rem;font-weight:900}.balance-label{font-size:.76rem;color:#75786f}.balance-vs{text-align:center;color:#a18454;font-size:1.6rem}
.action-box{background:linear-gradient(135deg,#ece5d5,#faf5e9);border:1px solid #d6c4a3;border-radius:20px;padding:1rem 1.05rem;margin-top:.9rem;box-shadow:inset 0 1px 0 rgba(255,255,255,.7)}.action-title{font-size:.84rem;color:#765c35;font-weight:850;margin-bottom:.45rem}
[data-testid="stMetric"]{background:#fffdf8;border:1px solid var(--line);padding:.65rem .75rem;border-radius:14px}[data-testid="stMetricValue"]{font-size:1.28rem;font-weight:850}.stButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:999px;min-height:2.7rem;font-weight:700}.stTextInput input{border-radius:13px!important}[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:rgba(255,253,248,.78)}
@media(max-width:700px){.hero-title{font-size:1.9rem}.hero-seal{width:115px;height:115px;right:-5px;top:10px;opacity:.19}.signal-grid{grid-template-columns:1fr}.balance-grid{grid-template-columns:1fr}.balance-vs{transform:rotate(90deg)}.block-container{padding-left:1rem;padding-right:1rem}.condition-head{display:block}.condition-meta{display:inline-block;margin-top:.35rem}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-center">
    <div class="hero-eyebrow">DIKE'S EYE · CONDITIONAL JUDGMENT</div>
    <div class="hero-title">평균의 진실이 아니라,<br><span class="gold">당신에게 유효한 진실을 판별합니다.</span></div>
    <div class="hero-sub">흩어진 후기와 엇갈린 의견을 한쪽으로 몰아가지 않습니다. 전체 여론, 당신이 중요하게 보는 조건, 상황에 따라 달라지는 평가, 그리고 리뷰에 남지 않은 신호를 저울 위에 함께 올려 선택을 돕습니다.</div>
    <div class="hero-chips"><span class="hero-chip">전체 여론</span><span class="hero-chip">조건부 진실</span><span class="hero-chip">상반된 증거</span><span class="hero-chip">Dike의 판정</span></div>
  </div>
  <svg class="hero-seal" viewBox="0 0 180 180" aria-hidden="true">
    <circle cx="90" cy="90" r="58" fill="none" stroke="#d9bd82" stroke-width="1.8"/>
    <path d="M90 52v55M64 70h52M70 70l-15 27h30L70 70zm40 0-15 27h30L110 70zM74 116h32" fill="none" stroke="#d9bd82" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
    <g fill="#879064"><ellipse cx="45" cy="118" rx="5" ry="12" transform="rotate(-38 45 118)"/><ellipse cx="38" cy="105" rx="5" ry="12" transform="rotate(-52 38 105)"/><ellipse cx="36" cy="90" rx="5" ry="12" transform="rotate(-68 36 90)"/><ellipse cx="40" cy="75" rx="5" ry="12" transform="rotate(-82 40 75)"/><ellipse cx="49" cy="62" rx="5" ry="12" transform="rotate(-102 49 62)"/><ellipse cx="135" cy="118" rx="5" ry="12" transform="rotate(38 135 118)"/><ellipse cx="142" cy="105" rx="5" ry="12" transform="rotate(52 142 105)"/><ellipse cx="144" cy="90" rx="5" ry="12" transform="rotate(68 144 90)"/><ellipse cx="140" cy="75" rx="5" ry="12" transform="rotate(82 140 75)"/><ellipse cx="131" cy="62" rx="5" ry="12" transform="rotate(102 131 62)"/></g>
    <path d="M52 132c18 18 58 18 76 0" fill="none" stroke="#879064" stroke-width="2"/>
  </svg>
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
    st.markdown('<div class="section-sub">대상과 조건을 평소 말하듯 적어주세요. “비싸도 괜찮지만 아늑해야 해”, “웨이팅은 싫어” 같은 허용·회피 조건까지 함께 읽습니다.</div>', unsafe_allow_html=True)
    st.caption("예: 야키니쿠 하코 어때? 가격은 좀 있어도 괜찮고, 안락함과 분위기가 중요해")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("질문", placeholder="대상 + 내 상황 + 중요조건", label_visibility="collapsed")
        submitted = st.form_submit_button("Dike에게 판단 맡기기", type="primary", use_container_width=True)

if submitted and question.strip():
    try:
        with st.spinner("질문의 조건과 허용 범위를 읽고 있어요..."):
            reset_after_question(parse_question(question.strip()))
    except Exception as exc:
        st.session_state.last_error = f"{type(exc).__name__}: {exc}"

intent = st.session_state.intent
if intent:
    kind = intent.get("kind", "restaurant")
    kind_label = "식당" if kind == "restaurant" else "상품"
    st.divider()
    st.markdown('<div class="section-title">Dike가 질문을 이렇게 이해했어요</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns([2.5, 1])
        with c1:
            st.markdown(f"**{kind_label} · {html.escape(str(intent.get('target','')))}**")
            st.caption(f"조건 해석 · {'LLM' if intent.get('parser_source') == 'openai' else '규칙 기반 fallback'}")
        with c2:
            st.metric("해석 신뢰도", f"{int(float(intent.get('parse_confidence',0))*100)}%")
        render_context_chips(current_context(), "질문에서 읽은 조건")
    if kind == "restaurant" and not st.session_state.selected_target:
        if st.button("NAVER에서 장소 확인", type="primary", use_container_width=True):
            try:
                with st.spinner("장소를 확인하고 있어요..."):
                    candidates, status = local_search(intent.get("target") or intent.get("original", ""), **naver_credentials())
                st.session_state.search_status = status
                st.session_state.candidates = candidates or [{"title":intent.get("target") or intent.get("original", ""),"category":"직접 입력","address":"","fallback":True}]
            except Exception as exc:
                st.session_state.last_error = f"{type(exc).__name__}: {exc}"
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
    st.markdown('<div class="section-title">이 조건을 저울에 올릴게요</div>', unsafe_allow_html=True)
    render_context_chips(current_context(), "분석 기준")
    with st.expander("조건이 다르면 수정"):
        a, b = st.columns(2)
        with a:
            st.text_input("요일/사용 상황", key="ctx_day")
            st.text_input("시간", key="ctx_time")
        with b:
            st.text_input("목적", key="ctx_purpose")
            st.text_input("중요 조건", key="ctx_preference")
    if st.button("이 조건으로 판단하기", type="primary", use_container_width=True):
        context = finalize_context(selected["name"])
        try:
            with st.spinner("전체 여론과 조건별 Evidence를 함께 저울질하고 있어요..."):
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
    consensus = a.get("consensus", {})
    conflicts = a.get("conflict_insights", [])
    condition_evidence = {str(x.get("aspect")): x for x in a.get("condition_evidence", [])}

    st.divider()
    st.markdown(f"<div class='result-hero'><div class='result-label'>DIKE'S JUDGMENT</div><div class='result-title'>{html.escape(str(r.get('headline','')))}</div><div class='result-summary'>{html.escape(str(r.get('summary','')))}</div></div>", unsafe_allow_html=True)
    render_context_chips(result_context, "이번 판정에 사용한 조건")
    s1, s2, s3 = st.columns(3)
    s1.markdown(f"<div class='stat-card'><div class='stat-label'>최종 적합도</div><div class='stat-value'>{d.get('fit_score',0):.0f}/100</div><div class='stat-note'>전체 여론 + 내 조건</div></div>", unsafe_allow_html=True)
    s2.markdown(f"<div class='stat-card'><div class='stat-label'>판단 신뢰도</div><div class='stat-value'>{d.get('confidence',0):.0f}%</div><div class='stat-note'>Evidence 품질·범위</div></div>", unsafe_allow_html=True)
    s3.markdown(f"<div class='stat-card'><div class='stat-label'>검토 Evidence</div><div class='stat-value'>{len(rows)+len(hidden_rows)}건</div><div class='stat-note'>Visible {len(rows)} + Hidden {len(hidden_rows)}</div></div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.2rem">먼저, 대체적인 전체 여론</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">긍정/부정 전용 검색어를 제외한 일반 후기 중심 표본입니다. 모집단 전체의 여론조사가 아니라, 현재 공개 검색으로 확보된 일반 후기의 방향입니다.</div>', unsafe_allow_html=True)
    total = int(consensus.get("sample_count", 0)); pos = int(consensus.get("positive_count", 0)); neg = int(consensus.get("negative_count", 0))
    pos_rate = float(consensus.get("positive_rate", 0)) * 100; neg_rate = float(consensus.get("negative_rate", 0)) * 100; opinion_score = float(consensus.get("opinion_score", 50))
    st.markdown(f"<div class='consensus-card'><div class='mini-eyebrow'>GENERAL CONSENSUS</div><div class='condition-name'>일반 후기 {total}건에서 본 전체적인 방향</div></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("긍정", f"{pos_rate:.0f}%", f"{pos}/{total}건" if total else "0건"); c2.metric("부정", f"{neg_rate:.0f}%", f"{neg}/{total}건" if total else "0건"); c3.metric("전체 여론 점수", f"{opinion_score:.0f}/100")
    bar_row("일반 후기 긍정", pos_rate, f"{pos}건 · {pos_rate:.0f}%", "positive")
    st.markdown("**긍정 쪽에서 자주 나온 말**"); st.markdown(f"<div class='kw-row'>{keyword_pills(consensus.get('positive_keywords', []),'positive')}</div>", unsafe_allow_html=True)
    st.markdown("**부정 쪽에서 자주 나온 말**"); st.markdown(f"<div class='kw-row'>{keyword_pills(consensus.get('negative_keywords', []),'negative')}</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.25rem">당신이 중요하게 본 조건</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">각 조건은 전체 관련 Evidence로 먼저 평가합니다. 요일·시간·목적과 맞는 표본이 충분하면 그 상황에서 실제로 얼마나 달라지는지도 별도로 보여줍니다.</div>', unsafe_allow_html=True)
    if condition_results:
        for item in condition_results:
            aspect = str(item.get("aspect") or ""); detail = condition_evidence.get(aspect, {}); label = str(item.get("label") or aspect_label(aspect))
            total = int(item.get("total_count", 0)); pos = int(item.get("positive_count", 0)); neg = int(item.get("negative_count", 0)); pos_rate = float(item.get("positive_rate", 0)) * 100; neg_rate = float(item.get("negative_rate", 0)) * 100; direct = int(item.get("direct_count", 0)); fit = float(item.get("fit", 0.5)) * 100; importance = float(item.get("importance", 0.8)) * 100; direction = direction_label(str(item.get("direction")))
            st.markdown(f"<div class='condition-card'><div class='condition-head'><div><div class='condition-name'>🎯 {html.escape(label)}</div><div class='condition-summary'>{html.escape(str(item.get('raw') or label))} · 관련 의견 {total}건 · 직접 검색/표현 근거 {direct}건</div></div><div class='condition-meta'>{html.escape(direction)} · 중요도 {importance:.0f}%</div></div></div>", unsafe_allow_html=True)
            if total >= 3:
                c1, c2, c3 = st.columns(3); c1.metric("긍정", f"{pos_rate:.0f}%", f"{pos}/{total}건"); c2.metric("부정", f"{neg_rate:.0f}%", f"{neg}/{total}건"); c3.metric("조건 적합", f"{fit:.0f}/100")
                bar_row("긍정 Evidence", pos_rate, f"{pos}건 · {pos_rate:.0f}%", "positive")
                st.markdown("**긍정 근거 키워드**"); st.markdown(f"<div class='kw-row'>{keyword_pills(detail.get('positive_keywords', []),'positive')}</div>", unsafe_allow_html=True)
                st.markdown("**부정 근거 키워드**"); st.markdown(f"<div class='kw-row'>{keyword_pills(detail.get('negative_keywords', []),'negative')}</div>", unsafe_allow_html=True)
                situ_count = int(item.get("situational_count", 0))
                if situ_count >= 3:
                    situ_neg = float(item.get("situational_negative_rate", 0)) * 100; lift = float(item.get("situational_lift", 0)) * 100
                    st.info(f"내 사용 상황과 직접 맞는 {situ_count}건에서는 부정 {situ_neg:.0f}%로, 이 조건 전체 평균보다 {lift:+.1f}%p 달랐습니다.")
                with st.expander(f"{label} 실제 근거 보기"):
                    evidence_samples("긍정 Evidence 예시", detail.get("positive_samples", [])); evidence_samples("부정 Evidence 예시", detail.get("negative_samples", []))
            else:
                st.warning(f"{label} 관련 의견 Evidence가 {total}건이라 안정적인 조건 판단에는 부족합니다.")
    else:
        st.info("질문에서 별도의 중요조건을 찾지 못해 전체 여론 중심으로 판단했습니다.")

    if conflicts:
        st.markdown('<div class="section-title" style="margin-top:1.25rem">의견이 갈린 핵심 쟁점</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">같은 대상을 두고 사람들이 왜 다르게 말하는지, 찬성과 반대 쪽에서 반복되는 단어와 실제 Evidence를 나눠봅니다.</div>', unsafe_allow_html=True)
        for conflict in conflicts[:3]:
            with st.container(border=True):
                label = str(conflict.get("label") or ""); pc = int(conflict.get("positive_count", 0)); nc = int(conflict.get("negative_count", 0)); total = max(1, pc + nc)
                st.markdown(f"**🎭 {html.escape(label)}** · 긍정 {pc}건 vs 부정 {nc}건"); bar_row("긍정", pc / total * 100, f"{pc}건", "positive"); bar_row("부정", nc / total * 100, f"{nc}건", "negative")
                lcol, rcol = st.columns(2)
                with lcol:
                    st.markdown("**긍정 쪽 핵심어**"); st.markdown(f"<div class='kw-row'>{keyword_pills(conflict.get('positive_keywords', []),'positive')}</div>", unsafe_allow_html=True); evidence_samples("긍정 Evidence", conflict.get("positive_samples", []))
                with rcol:
                    st.markdown("**부정 쪽 핵심어**"); st.markdown(f"<div class='kw-row'>{keyword_pills(conflict.get('negative_keywords', []),'negative')}</div>", unsafe_allow_html=True); evidence_samples("부정 Evidence", conflict.get("negative_samples", []))

    signal_counts = a.get("wald", {}).get("signal_counts", {})
    if signal_counts:
        st.markdown('<div class="section-title" style="margin-top:1.2rem">리뷰만 보면 놓칠 수 있는 신호</div>', unsafe_allow_html=True)
        with st.container(border=True):
            top = sorted(signal_counts.items(), key=lambda x:x[1], reverse=True)[:3]; cards = "".join(f"<div class='signal-card'><div class='signal-label'>{html.escape(WALD_LABELS.get(k,k))}</div><div class='signal-value'>{v}건</div></div>" for k,v in top)
            st.markdown(f"<div class='signal-grid'>{cards}</div>", unsafe_allow_html=True); st.caption("실제 발생률이 아니라 검색된 이탈·실패 신호의 건수입니다.")

    st.markdown('<div class="section-title" style="margin-top:1.25rem">⚖️ Dike의 저울</div>', unsafe_allow_html=True)
    overall_score = float(comps.get("consensus_score", 50)); condition_score = float(comps.get("condition_score", 50))
    st.markdown(f"<div class='balance-card'><div class='mini-eyebrow'>BALANCE OF TRUTH</div><div class='balance-grid'><div class='balance-side'><div class='balance-label'>전체 여론</div><div class='balance-score'>{overall_score:.0f}</div><div class='balance-label'>일반 후기 중심</div></div><div class='balance-vs'>⚖</div><div class='balance-side'><div class='balance-label'>나에게 유효한 진실</div><div class='balance-score'>{condition_score:.0f}</div><div class='balance-label'>조건 중요도 반영</div></div></div></div>", unsafe_allow_html=True)
    delta = condition_score - overall_score
    if abs(delta) < 7: st.caption("전체 여론과 내 조건의 결론이 대체로 같은 방향입니다.")
    elif delta > 0: st.caption(f"전체 평판보다 내 조건에서 약 {delta:.0f}점 더 유리합니다. 평균보다 나에게 더 잘 맞는 선택일 수 있습니다.")
    else: st.caption(f"전체 평판보다 내 조건에서 약 {abs(delta):.0f}점 불리합니다. 대체로 좋은 평가여도 나에게는 조건부일 수 있습니다.")

    st.markdown('<div class="action-box"><div class="action-title">🏛️ Dike의 최종 판정 · Solomon Choice</div>', unsafe_allow_html=True)
    for item in r.get("recommendations", []): st.markdown(f"- **{item}**")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("AI가 판정 이유를 더 자세히 설명하기"):
        st.caption("점수·건수·비율은 계산 엔진이 확정하고, AI는 그 근거를 읽기 쉽게 정리합니다.")
        if st.button("설명 생성", use_container_width=True):
            with st.spinner("전체 여론과 조건별 Evidence를 함께 정리하고 있어요..."):
                st.session_state.explanation = generate_explanation(a, api_key=secret("OPENAI_API_KEY"), model=secret("OPENAI_MODEL", "gpt-5-mini"))
        if st.session_state.explanation:
            e = st.session_state.explanation; st.markdown(f"**{e.get('headline','')}**"); st.write(e.get("answer", ""))
            for reason in e.get("reasons", []): st.markdown(f"- {reason}")

    with st.expander("분석 근거 자세히 보기"):
        st.markdown("##### 전체 여론"); st.json(consensus)
        st.markdown("##### 조건별 계산"); cdf = pd.DataFrame(condition_results)
        if not cdf.empty:
            cols = [c for c in ["label","raw","direction","importance","total_count","positive_count","negative_count","direct_count","situational_count","situational_lift","fit","evidence_confidence"] if c in cdf.columns]; st.dataframe(cdf[cols], use_container_width=True, hide_index=True)
        st.markdown("##### 점수 구성"); st.json({"consensus_score":comps.get("consensus_score"),"condition_score":comps.get("condition_score"),"weighted_sentiment":comps.get("weighted_sentiment"),"condition_fit":comps.get("condition_fit"),"condition_coverage":comps.get("condition_coverage"),"positive_strength":comps.get("positive_strength"),"rca_risk":comps.get("rca_risk"),"wald_risk":comps.get("wald_risk"),"policy":d.get("policy",{})})
        st.markdown("##### 상위 Evidence"); df = pd.DataFrame(rows); cols = [c for c in ["source","retrieval_scope","title","aspects","contexts","condition_direct_aspects","situational_aligned","sentiment","R","F","M","priority"] if c in df.columns]
        if not df.empty: st.dataframe(df[cols].head(40), use_container_width=True, hide_index=True)

    if st.button("새로운 질문하기", use_container_width=True):
        for key in list(DEFAULTS) + ["ctx_day","ctx_time","ctx_purpose","ctx_preference","target_edit"]: st.session_state.pop(key, None)
        st.rerun()

if st.session_state.last_error:
    st.error("처리 중 문제가 생겼어요. 입력한 내용은 유지되어 있습니다.")
    with st.expander("오류 정보"): st.code(st.session_state.last_error)

st.divider()
st.caption("Dike's Eye · 전체 여론과 조건부 진실을 함께 저울질하고, Evidence·비율·점수는 deterministic engine이 계산합니다.")

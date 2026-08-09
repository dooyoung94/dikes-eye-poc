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

st.set_page_config(page_title="Dike's Eye", page_icon="⚖️", layout="centered", initial_sidebar_state="collapsed")


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, os.getenv(name, default))
        return str(value) if value is not None else default
    except Exception:
        return os.getenv(name, default)


def parse_question(text: str) -> dict:
    try:
        return parse_intent(text, api_key=secret("OPENAI_API_KEY"), model=secret("OPENAI_MODEL", "gpt-5-mini"))
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
    ctx = intent_context(intent)
    st.session_state.intent = intent
    st.session_state.parsed_context = ctx
    st.session_state.candidates = []
    st.session_state.selected_target = None
    st.session_state.analysis = None
    st.session_state.user_report = None
    st.session_state.explanation = None
    st.session_state.last_error = ""
    st.session_state.ctx_day = ctx["date_or_day"]
    st.session_state.ctx_time = ctx["time"]
    st.session_state.ctx_purpose = ctx["purpose"]
    st.session_state.ctx_preference = ctx["preference"]
    st.session_state.target_edit = str(intent.get("target") or "")


def current_context() -> dict:
    parsed = st.session_state.get("parsed_context") or intent_context(st.session_state.get("intent"))
    values = {
        "date_or_day": str(st.session_state.get("ctx_day") or parsed.get("date_or_day") or "").strip(),
        "time": str(st.session_state.get("ctx_time") or parsed.get("time") or "").strip(),
        "purpose": str(st.session_state.get("ctx_purpose") or parsed.get("purpose") or "").strip(),
        "preference": str(st.session_state.get("ctx_preference") or parsed.get("preference") or "").strip(),
    }
    if values["preference"] == str(parsed.get("preference") or "").strip():
        values["conditions"] = parsed.get("conditions", [])
    else:
        values["conditions"] = normalize_context_conditions({"preference": values["preference"]})
    return values


def finalize_context(target: str) -> dict:
    context = current_context()
    parsed = st.session_state.get("parsed_context") or {}
    if context.get("preference") and context["preference"] != str(parsed.get("preference") or "").strip():
        try:
            reparsed = parse_question(f"{target} 어때? {context['preference']}")
            if reparsed.get("conditions"):
                context["conditions"] = reparsed["conditions"]
        except Exception:
            pass
    return context


def direction_label(direction: str) -> str:
    return {"prefer": "중요", "avoid": "회피", "tolerate": "허용"}.get(str(direction), "중요")


def render_context_chips(context: dict, title: str) -> None:
    chips = []
    for label, value in [("상황", context.get("date_or_day")), ("시간", context.get("time")), ("목적", context.get("purpose"))]:
        if str(value or "").strip():
            chips.append(f"<span class='chip'><small>{html.escape(label)}</small><b>{html.escape(str(value))}</b></span>")
    for cond in normalize_context_conditions(context):
        raw = str(cond.get("raw") or cond.get("label") or "")
        chips.append(f"<span class='chip condition'><small>{direction_label(str(cond.get('direction')))}</small><b>{html.escape(raw)}</b></span>")
    body = "".join(chips) if chips else "<span class='muted'>별도 조건 없음</span>"
    st.markdown(f"<div class='context'><div class='context-title'>◎ {html.escape(title)}</div><div class='chips'>{body}</div></div>", unsafe_allow_html=True)


def run_decision(target: str, kind: str, context: dict) -> dict:
    creds = naver_credentials()
    visible_raw = collect_visible_evidence(target, kind=kind, context=context, **creds)
    hidden_raw = collect_hidden_evidence(target, kind=kind, context=context, **creds)
    visible_norm = normalize_evidence(visible_raw, context)
    hidden_norm = normalize_evidence(hidden_raw, context)
    rows, rfm_summary = build_rfm(visible_norm)
    eda = build_eda(rows)
    rca = derive_rca(rows, context)
    rashomon = build_rashomon(rca)
    wald = analyze_wald(hidden_norm, kind=kind)
    consensus = build_consensus(rows)
    conflicts = build_conflict_insights(rows, rca.get("conflicts", []))
    condition_evidence = build_condition_evidence(rows, rca.get("condition_results", []))
    decision = score_decision(rows, eda, rfm_summary, rca, wald, consensus=consensus)
    return {"kind": kind, "target": target, "context": dict(context), "rows": rows, "hidden_rows": hidden_norm,
            "rfm": rfm_summary, "eda": eda, "rca": rca, "rashomon": rashomon, "wald": wald,
            "consensus": consensus, "conflict_insights": conflicts, "condition_evidence": condition_evidence,
            "decision": decision}


def bar(label: str, value: float, text: str, tone: str = "gold") -> None:
    width = max(0, min(100, float(value)))
    st.markdown(f"<div class='bar'><div><span>{html.escape(label)}</span><b>{html.escape(text)}</b></div><i><em class='{tone}' style='width:{width:.1f}%'></em></i></div>", unsafe_allow_html=True)


def pills(items: list[dict], tone: str) -> str:
    if not items:
        return "<span class='muted'>반복 키워드 없음</span>"
    return "".join(f"<span class='kw {tone}'>{html.escape(str(x.get('keyword','')))} <b>{int(x.get('count',0))}</b></span>" for x in items[:5])


def evidence_samples(title: str, samples: list[dict]) -> None:
    if not samples:
        return
    st.markdown(f"**{title}**")
    for x in samples[:2]:
        st.markdown(f"<div class='quote'><small>{html.escape(str(x.get('source') or 'Evidence'))}</small><b>{html.escape(str(x.get('title') or x.get('evidence_id') or ''))}</b><p>{html.escape(str(x.get('snippet') or ''))}</p></div>", unsafe_allow_html=True)


st.markdown("""
<style>
:root{--paper:#fffdf8;--ink:#252722;--sub:#6e7069;--gold:#b9965d;--deepgold:#80633c;--olive:#737b59;--navy:#263746;--navy2:#354b5c;--line:#ddd2bd;--green:#6d8d73;--red:#aa6662}
html,body,[class*="css"]{font-family:Pretendard,-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Segoe UI",sans-serif}
[data-testid="stAppViewContainer"]{position:relative;color:var(--ink);background:linear-gradient(rgba(249,246,239,.91),rgba(243,237,226,.94)),radial-gradient(circle at 50% 8%,#fffdf8 0,#eee4d3 78%);background-attachment:fixed}
[data-testid="stAppViewContainer"]:before{content:"";position:fixed;z-index:0;pointer-events:none;left:0;right:0;top:70px;height:520px;opacity:.115;background-repeat:no-repeat;background-position:center top;background-size:min(1180px,130vw) auto;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 520'%3E%3Cg fill='none' stroke='%2384765e' stroke-width='4'%3E%3Cpath d='M80 410H1120M150 380H1050M210 350H990'/%3E%3Cpath d='M250 350V180M310 350V180M890 350V180M950 350V180'/%3E%3Cpath d='M225 180h110l-55-55zM865 180h110l-55-55z'/%3E%3Cpath d='M390 350V135M455 350V135M520 350V135M585 350V135M650 350V135M715 350V135M780 350V135'/%3E%3Cpath d='M360 135h450L585 48zM345 350h480'/%3E%3Cellipse cx='585' cy='420' rx='390' ry='70'/%3E%3C/g%3E%3C/svg%3E")}
[data-testid="stAppViewContainer"]:after{content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.20;background-image:linear-gradient(110deg,transparent 48%,rgba(117,101,76,.045) 49%,transparent 50%),linear-gradient(70deg,transparent 72%,rgba(117,101,76,.035) 73%,transparent 74%);background-size:310px 240px,390px 300px}
[data-testid="stHeader"]{background:transparent}.block-container{position:relative;z-index:1;max-width:860px;padding-top:1rem;padding-bottom:5rem}
.hero{position:relative;overflow:hidden;border-radius:30px;padding:2.25rem 2.1rem 1.8rem;margin:.25rem 0 1rem;color:#fbf7ef;background:radial-gradient(circle at 78% 18%,rgba(200,175,121,.14),transparent 26%),linear-gradient(145deg,var(--navy) 0%,#2e4354 54%,#1f303e 100%);border:1px solid rgba(213,185,128,.58);box-shadow:0 20px 52px rgba(38,48,54,.20),inset 0 0 0 5px rgba(255,255,255,.025)}
.hero:before{content:"";position:absolute;inset:9px;border:1px solid rgba(210,180,120,.28);border-radius:23px;pointer-events:none}.hero:after{content:"";position:absolute;left:12%;right:12%;bottom:18px;height:1px;background:linear-gradient(90deg,transparent,#c7a66b,transparent);opacity:.72}
.hero-mark{position:absolute;right:1.2rem;top:.9rem;width:150px;height:150px;opacity:.34}.hero-copy{position:relative;z-index:2;max-width:610px}.eyebrow{font-size:.69rem;font-weight:850;letter-spacing:.18em;color:#d6bd8b}.hero h1{font-size:2.35rem;line-height:1.24;letter-spacing:-.045em;margin:.85rem 0 .7rem;font-weight:850}.hero h1 span{color:#e1c58f}.hero p{font-size:.94rem;line-height:1.72;color:#e6e8e5;margin:0;max-width:590px}.hero-tags{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:1rem;padding-bottom:.45rem}.hero-tags span{font-size:.7rem;letter-spacing:.03em;padding:.32rem .62rem;border:1px solid rgba(220,195,145,.3);border-radius:999px;background:rgba(255,255,255,.045);color:#eee9df}
.section-title{font-size:1.22rem;font-weight:850;letter-spacing:-.025em;margin:.35rem 0 .2rem}.section-sub{font-size:.86rem;line-height:1.55;color:var(--sub);margin-bottom:.7rem}.panel,.result,.condition-card,.balance,.verdict{background:rgba(255,253,248,.94);border:1px solid var(--line);border-radius:20px;padding:1rem;box-shadow:0 8px 24px rgba(68,56,37,.055);backdrop-filter:blur(3px)}
.result{border-top:3px solid var(--gold)}.mini{font-size:.68rem;font-weight:850;letter-spacing:.13em;color:var(--deepgold)}.result h2{font-size:1.55rem;letter-spacing:-.035em;margin:.28rem 0}.result p{font-size:.91rem;line-height:1.6;color:#5d615a}.context{background:rgba(251,247,238,.94);border:1px solid #d9c9aa;border-radius:17px;padding:.78rem .82rem;margin:.65rem 0}.context-title{font-size:.76rem;font-weight:850;color:#735c37;margin-bottom:.45rem}.chips{display:flex;gap:.35rem;flex-wrap:wrap}.chip{display:inline-flex;gap:.28rem;align-items:center;background:#fffdf8;border:1px solid #dfd3be;border-radius:999px;padding:.32rem .55rem;font-size:.73rem}.chip small{color:#8a8b84}.chip b{color:#292b27}.chip.condition{border-color:#cdbb93;background:#fff8e9}
.stat{background:rgba(255,253,248,.96);border:1px solid var(--line);border-radius:15px;padding:.72rem}.stat small{color:var(--sub);font-weight:700}.stat strong{display:block;font-size:1.3rem;margin-top:.08rem}.stat em{font-size:.7rem;color:#85877f;font-style:normal}.condition-card{margin:.6rem 0}.condition-head{display:flex;justify-content:space-between;gap:.5rem}.condition-head strong{font-size:1rem}.badge{font-size:.7rem;color:#725b38;background:#f3ead9;border-radius:999px;padding:.25rem .48rem;white-space:nowrap}.condition-note{font-size:.78rem;color:#70736c;margin-top:.25rem}.bar{margin:.5rem 0}.bar>div{display:flex;justify-content:space-between;font-size:.76rem;margin-bottom:.2rem;color:#62655f}.bar i{display:block;height:8px;border-radius:99px;background:#ebe5da;overflow:hidden}.bar em{display:block;height:100%;border-radius:99px}.bar em.positive{background:var(--green)}.bar em.negative{background:var(--red)}.bar em.gold{background:var(--gold)}
.kwrow{display:flex;flex-wrap:wrap;gap:.3rem;margin:.35rem 0 .65rem}.kw{font-size:.7rem;border-radius:999px;padding:.27rem .48rem;border:1px solid}.kw.positive{background:#f0f5ef;border-color:#c9d8c9;color:#4d6753}.kw.negative{background:#f8eeee;border-color:#e4c9c8;color:#80504d}.muted{font-size:.73rem;color:#8c8e87}.quote{border-left:3px solid #b99b67;background:#faf6ed;border-radius:0 11px 11px 0;padding:.6rem .68rem;margin:.38rem 0}.quote small{display:block;color:#8b7e68;font-weight:800;font-size:.63rem}.quote b{display:block;font-size:.78rem;margin:.1rem 0}.quote p{font-size:.75rem;line-height:1.45;color:#62645f;margin:0}.signals{display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem}.signal{background:#faf6ee;border:1px solid var(--line);border-radius:13px;padding:.65rem}.signal small{color:var(--sub);display:block}.signal b{font-size:1.05rem}.balance{border-top:3px solid var(--gold)}.balance-grid{display:grid;grid-template-columns:1fr 52px 1fr;align-items:center;gap:.5rem;margin-top:.6rem}.side{text-align:center;background:#faf6ed;border:1px solid #e2d7c2;border-radius:14px;padding:.72rem}.side small{color:#777970}.score{font-size:1.6rem;font-weight:900;color:#31342e}.scale{text-align:center;font-size:1.45rem;color:#a48655}.verdict{margin-top:.8rem;border:2px solid #c8a66c;box-shadow:inset 0 0 0 4px #f7efdf,0 10px 26px rgba(82,61,29,.08)}.verdict-title{font-size:.75rem;font-weight:900;letter-spacing:.08em;color:#76582e;margin-bottom:.45rem}
[data-testid="stMetric"]{background:#fffdf8;border:1px solid var(--line);padding:.55rem .62rem;border-radius:13px}[data-testid="stMetricValue"]{font-size:1.18rem;font-weight:850}.stButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:999px;min-height:2.7rem;font-weight:750}.stTextInput input{border-radius:13px!important}[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:rgba(255,253,248,.88)}
@media(max-width:700px){[data-testid="stAppViewContainer"]:before{top:95px;height:390px;opacity:.09;background-size:820px auto}.block-container{padding:.55rem .78rem 4.5rem}.hero{border-radius:22px;padding:1.35rem 1.15rem 1.2rem;margin-top:.1rem}.hero:before{inset:6px;border-radius:17px}.hero-mark{width:92px;height:92px;right:-.15rem;top:.3rem;opacity:.25}.eyebrow{font-size:.57rem;letter-spacing:.13em}.hero h1{font-size:1.62rem;line-height:1.3;margin:.65rem 0 .55rem;max-width:92%}.hero p{font-size:.79rem;line-height:1.6;max-width:95%}.hero-tags{gap:.28rem;margin-top:.75rem}.hero-tags span{font-size:.62rem;padding:.27rem .45rem}.section-title{font-size:1.08rem}.section-sub{font-size:.78rem}.panel,.result,.condition-card,.balance,.verdict{border-radius:16px;padding:.82rem}.result h2{font-size:1.35rem}.result p{font-size:.82rem}.chips{gap:.28rem}.chip{font-size:.67rem;padding:.28rem .45rem}.condition-head{display:block}.badge{display:inline-block;margin-top:.28rem}.signals{grid-template-columns:1fr}.balance-grid{grid-template-columns:1fr 38px 1fr;gap:.3rem}.side{padding:.58rem .3rem}.score{font-size:1.35rem}.scale{font-size:1.15rem}.quote p{font-size:.72rem}[data-testid="column"]{min-width:0!important}.stButton>button,[data-testid="stFormSubmitButton"]>button{min-height:2.85rem}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='hero'>
 <div class='hero-copy'>
  <div class='eyebrow'>DIKE'S EYE · CONDITIONAL JUDGMENT</div>
  <h1>평균의 진실이 아니라,<br><span>당신에게 유효한 진실을 판별합니다.</span></h1>
  <p>전체 여론, 당신의 조건, 서로 엇갈린 증거와 리뷰 밖의 신호를 하나의 저울에 올립니다. 디케는 가장 큰 목소리가 아니라, 이번 선택에 가장 유효한 근거를 판별합니다.</p>
  <div class='hero-tags'><span>CONSENSUS</span><span>CONDITION</span><span>EVIDENCE</span><span>JUDGMENT</span></div>
 </div>
 <svg class='hero-mark' viewBox='0 0 180 180' aria-hidden='true'><g fill='none' stroke='#d8b978' stroke-width='2'><path d='M90 45v70M58 68h64M65 68L48 98h34L65 68zm50 0L98 98h34l-17-30zM70 120h40'/><path d='M45 132c-20-20-20-54 1-75M135 132c20-20 20-54-1-75'/></g><g fill='#7f8964'><ellipse cx='42' cy='118' rx='5' ry='12' transform='rotate(-35 42 118)'/><ellipse cx='35' cy='100' rx='5' ry='12' transform='rotate(-55 35 100)'/><ellipse cx='36' cy='80' rx='5' ry='12' transform='rotate(-75 36 80)'/><ellipse cx='138' cy='118' rx='5' ry='12' transform='rotate(35 138 118)'/><ellipse cx='145' cy='100' rx='5' ry='12' transform='rotate(55 145 100)'/><ellipse cx='144' cy='80' rx='5' ry='12' transform='rotate(75 144 80)'/></g></svg>
</div>
""", unsafe_allow_html=True)

DEFAULTS = {"intent": None, "parsed_context": {}, "candidates": [], "selected_target": None, "analysis": None, "user_report": None, "explanation": None, "last_error": ""}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.container(border=True):
    st.markdown("<div class='section-title'>어떤 선택을 고민하고 있나요?</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>대상과 조건을 평소 말하듯 적어주세요. 허용할 것과 피하고 싶은 조건까지 함께 읽습니다.</div>", unsafe_allow_html=True)
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("질문", placeholder="예: 야키니쿠 하코 어때? 비싸도 괜찮지만 조용하고 편안했으면 해", label_visibility="collapsed")
        submitted = st.form_submit_button("Dike에게 판단 맡기기", type="primary", use_container_width=True)

if submitted and question.strip():
    try:
        with st.spinner("질문에서 조건과 허용 범위를 읽고 있어요..."):
            reset_after_question(parse_question(question.strip()))
    except Exception as exc:
        st.session_state.last_error = f"{type(exc).__name__}: {exc}"

intent = st.session_state.intent
if intent:
    kind = intent.get("kind", "restaurant")
    kind_label = "식당" if kind == "restaurant" else "상품"
    st.markdown("<div class='section-title'>Dike가 질문을 이렇게 이해했어요</div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(f"**{kind_label} · {html.escape(str(intent.get('target','')))}**")
        st.caption(f"조건 해석 · {'LLM' if intent.get('parser_source') == 'openai' else '규칙 기반 fallback'} · 신뢰도 {int(float(intent.get('parse_confidence',0))*100)}%")
        render_context_chips(current_context(), "질문에서 읽은 조건")
    if kind == "restaurant" and not st.session_state.selected_target:
        if st.button("NAVER에서 장소 확인", type="primary", use_container_width=True):
            try:
                with st.spinner("장소를 확인하고 있어요..."):
                    candidates, _ = local_search(intent.get("target") or intent.get("original", ""), **naver_credentials())
                st.session_state.candidates = candidates or [{"title": intent.get("target") or intent.get("original", ""), "category": "직접 입력", "address": ""}]
            except Exception as exc:
                st.session_state.last_error = f"{type(exc).__name__}: {exc}"
        if st.session_state.get("candidates"):
            labels = [f"{x.get('title','')} · {x.get('category','')} · {x.get('address','')}" for x in st.session_state.candidates]
            idx = st.radio("이 장소가 맞나요?", range(len(labels)), format_func=lambda i: labels[i])
            if st.button("이 장소가 맞아요", type="primary", use_container_width=True):
                chosen = st.session_state.candidates[idx]
                st.session_state.selected_target = {"kind": "restaurant", "name": chosen.get("title") or intent.get("target", ""), "meta": chosen}
    elif kind == "product" and not st.session_state.selected_target:
        with st.form("product_confirm"):
            product = st.text_input("제품명 / 모델명", key="target_edit")
            if st.form_submit_button("이 제품이 맞아요", type="primary", use_container_width=True) and product.strip():
                st.session_state.selected_target = {"kind": "product", "name": product.strip(), "meta": {}}

selected = st.session_state.selected_target
if selected and not st.session_state.analysis:
    st.markdown("<div class='section-title'>이 조건을 저울에 올릴게요</div>", unsafe_allow_html=True)
    render_context_chips(current_context(), "분석 기준")
    with st.expander("조건이 다르면 수정"):
        st.text_input("요일/사용 상황", key="ctx_day")
        st.text_input("시간", key="ctx_time")
        st.text_input("목적", key="ctx_purpose")
        st.text_input("중요 조건", key="ctx_preference")
    if st.button("이 조건으로 판단하기", type="primary", use_container_width=True):
        try:
            context = finalize_context(selected["name"])
            with st.spinner("전체 여론과 조건별 Evidence를 저울질하고 있어요..."):
                analysis = run_decision(selected["name"], selected["kind"], context)
                report = build_user_report(analysis, selected["name"], selected["kind"])
            st.session_state.analysis, st.session_state.user_report = analysis, report
        except Exception as exc:
            st.session_state.last_error = f"{type(exc).__name__}: {exc}"

if st.session_state.analysis and st.session_state.user_report:
    a, r = st.session_state.analysis, st.session_state.user_report
    d = a["decision"]; comps = d.get("components", {}); rows = a.get("rows", []); hidden = a.get("hidden_rows", [])
    conditions = a.get("rca", {}).get("condition_results", []); consensus = a.get("consensus", {}); conflicts = a.get("conflict_insights", [])
    details = {str(x.get("aspect")): x for x in a.get("condition_evidence", [])}
    st.markdown(f"<div class='result'><div class='mini'>DIKE'S JUDGMENT</div><h2>{html.escape(str(r.get('headline','')))}</h2><p>{html.escape(str(r.get('summary','')))}</p></div>", unsafe_allow_html=True)
    render_context_chips(a.get("context", {}), "이번 판정 조건")
    c1, c2, c3 = st.columns(3)
    for col, label, value, note in [(c1,"최종 적합도",f"{d.get('fit_score',0):.0f}/100","전체+조건"),(c2,"판단 신뢰도",f"{d.get('confidence',0):.0f}%","근거 품질"),(c3,"Evidence",f"{len(rows)+len(hidden)}건","검토 표본")]:
        col.markdown(f"<div class='stat'><small>{label}</small><strong>{value}</strong><em>{note}</em></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ⅰ. 시민의 목소리 · 전체 여론</div><div class='section-sub'>긍정·부정 전용 검색을 제외한 일반 후기 중심으로 대체적인 방향을 봅니다.</div>", unsafe_allow_html=True)
    total = int(consensus.get("sample_count",0)); pos = int(consensus.get("positive_count",0)); neg = int(consensus.get("negative_count",0)); pr = float(consensus.get("positive_rate",0))*100; nr = float(consensus.get("negative_rate",0))*100
    c1,c2,c3 = st.columns(3); c1.metric("긍정",f"{pr:.0f}%",f"{pos}/{total}건"); c2.metric("부정",f"{nr:.0f}%",f"{neg}/{total}건"); c3.metric("여론 점수",f"{float(consensus.get('opinion_score',50)):.0f}/100")
    bar("일반 후기 긍정",pr,f"{pos}건 · {pr:.0f}%","positive")
    st.markdown(f"<div class='kwrow'>{pills(consensus.get('positive_keywords',[]),'positive')}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='kwrow'>{pills(consensus.get('negative_keywords',[]),'negative')}</div>",unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ⅱ. 당신의 조건</div><div class='section-sub'>평균이 좋아도 내 조건에서 달라질 수 있습니다. 조건별 표본·방향·적합도를 따로 계산합니다.</div>", unsafe_allow_html=True)
    if conditions:
        for item in conditions:
            aspect=str(item.get("aspect") or ""); detail=details.get(aspect,{}); label=str(item.get("label") or aspect_label(aspect)); total=int(item.get("total_count",0)); pos=int(item.get("positive_count",0)); neg=int(item.get("negative_count",0)); pr=float(item.get("positive_rate",0))*100; fit=float(item.get("fit",.5))*100
            st.markdown(f"<div class='condition-card'><div class='condition-head'><div><strong>◎ {html.escape(label)}</strong><div class='condition-note'>{html.escape(str(item.get('raw') or label))} · 관련 {total}건 · 직접 근거 {int(item.get('direct_count',0))}건</div></div><span class='badge'>{direction_label(str(item.get('direction')))} · 중요도 {float(item.get('importance',.8))*100:.0f}%</span></div></div>", unsafe_allow_html=True)
            if total >= 3:
                c1,c2,c3=st.columns(3); c1.metric("긍정",f"{pr:.0f}%",f"{pos}/{total}건"); c2.metric("부정",f"{100-pr:.0f}%",f"{neg}/{total}건"); c3.metric("조건 적합",f"{fit:.0f}/100")
                bar("긍정 Evidence",pr,f"{pos}건","positive")
                st.markdown(f"<div class='kwrow'>{pills(detail.get('positive_keywords',[]),'positive')}</div>",unsafe_allow_html=True)
                st.markdown(f"<div class='kwrow'>{pills(detail.get('negative_keywords',[]),'negative')}</div>",unsafe_allow_html=True)
                if int(item.get("situational_count",0)) >= 3:
                    st.info(f"내 상황과 직접 맞는 {int(item.get('situational_count',0))}건에서는 부정 {float(item.get('situational_negative_rate',0))*100:.0f}% · 전체 대비 {float(item.get('situational_lift',0))*100:+.1f}%p")
                with st.expander(f"{label} 실제 근거"):
                    evidence_samples("긍정 Evidence",detail.get("positive_samples",[])); evidence_samples("부정 Evidence",detail.get("negative_samples",[]))
            else:
                st.warning(f"{label} 관련 Evidence가 {total}건이라 판단 안정성이 낮습니다.")

    if conflicts:
        st.markdown("<div class='section-title'>Ⅲ. Rashomon · 갈라진 진실</div><div class='section-sub'>같은 대상을 두고 의견이 갈린 지점을 양쪽 근거로 나눠 봅니다.</div>", unsafe_allow_html=True)
        for x in conflicts[:3]:
            with st.container(border=True):
                pc=int(x.get("positive_count",0)); nc=int(x.get("negative_count",0)); t=max(1,pc+nc); st.markdown(f"**{html.escape(str(x.get('label') or '쟁점'))}** · 긍정 {pc}건 vs 부정 {nc}건")
                bar("긍정",pc/t*100,f"{pc}건","positive"); bar("부정",nc/t*100,f"{nc}건","negative")
                st.markdown(f"<div class='kwrow'>{pills(x.get('positive_keywords',[]),'positive')}</div>",unsafe_allow_html=True); st.markdown(f"<div class='kwrow'>{pills(x.get('negative_keywords',[]),'negative')}</div>",unsafe_allow_html=True)
                with st.expander("양쪽 실제 Evidence"):
                    evidence_samples("긍정 쪽",x.get("positive_samples",[])); evidence_samples("부정 쪽",x.get("negative_samples",[]))

    signals=a.get("wald",{}).get("signal_counts",{})
    if signals:
        st.markdown("<div class='section-title'>Ⅳ. Wald · 사라진 진실</div><div class='section-sub'>좋은 리뷰만으로는 보이지 않는 이탈·실패·불편 신호를 따로 확인합니다.</div>",unsafe_allow_html=True)
        top=sorted(signals.items(),key=lambda x:x[1],reverse=True)[:3]; cards="".join(f"<div class='signal'><small>{html.escape(WALD_LABELS.get(k,k))}</small><b>{v}건</b></div>" for k,v in top); st.markdown(f"<div class='signals'>{cards}</div>",unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ⅴ. Dike의 저울</div>",unsafe_allow_html=True)
    overall=float(comps.get("consensus_score",50)); personal=float(comps.get("condition_score",50)); delta=personal-overall
    st.markdown(f"<div class='balance'><div class='mini'>BALANCE OF TRUTH</div><div class='balance-grid'><div class='side'><small>전체 여론</small><div class='score'>{overall:.0f}</div></div><div class='scale'>⚖</div><div class='side'><small>나에게 유효한 진실</small><div class='score'>{personal:.0f}</div></div></div></div>",unsafe_allow_html=True)
    if abs(delta)<7: st.caption("전체 여론과 내 조건이 대체로 같은 방향입니다.")
    elif delta>0: st.caption(f"평균보다 내 조건에서 {delta:.0f}점 더 유리합니다. 대중적 평가보다 나에게 더 잘 맞을 수 있습니다.")
    else: st.caption(f"평균보다 내 조건에서 {abs(delta):.0f}점 불리합니다. 대체로 좋은 평가여도 이번 선택에는 조건이 붙습니다.")

    st.markdown("<div class='verdict'><div class='verdict-title'>Ⅵ. DIKE'S JUDGMENT · SOLOMON CHOICE</div>",unsafe_allow_html=True)
    for item in r.get("recommendations",[]): st.markdown(f"- **{item}**")
    st.markdown("</div>",unsafe_allow_html=True)

    with st.expander("AI가 판정 이유를 더 자세히 설명"):
        if st.button("설명 생성",use_container_width=True):
            with st.spinner("판정 근거를 정리하고 있어요..."):
                st.session_state.explanation=generate_explanation(a,api_key=secret("OPENAI_API_KEY"),model=secret("OPENAI_MODEL","gpt-5-mini"))
        if st.session_state.explanation:
            e=st.session_state.explanation; st.markdown(f"**{e.get('headline','')}**"); st.write(e.get("answer","")); [st.markdown(f"- {x}") for x in e.get("reasons",[])]
    with st.expander("분석 근거 자세히 보기"):
        st.json(consensus); df=pd.DataFrame(conditions)
        if not df.empty: st.dataframe(df,use_container_width=True,hide_index=True)
        st.json({"consensus_score":overall,"condition_score":personal,"policy":d.get("policy",{})})
    if st.button("새로운 질문하기",use_container_width=True):
        for key in list(DEFAULTS)+["ctx_day","ctx_time","ctx_purpose","ctx_preference","target_edit"]: st.session_state.pop(key,None)
        st.rerun()

if st.session_state.last_error:
    st.error("처리 중 문제가 생겼어요. 입력 내용은 유지되어 있습니다.")
    with st.expander("오류 정보"): st.code(st.session_state.last_error)

st.divider()
st.caption("Dike's Eye · Consensus × Condition × Rashomon × Wald → Judgment")

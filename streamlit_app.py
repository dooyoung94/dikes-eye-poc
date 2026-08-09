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
    for label, value in [
        ("상황", context.get("date_or_day")),
        ("시간", context.get("time")),
        ("목적", context.get("purpose")),
    ]:
        if str(value or "").strip():
            chips.append(
                f"<span class='chip'><small>{html.escape(label)}</small>"
                f"<b>{html.escape(str(value))}</b></span>"
            )
    for cond in normalize_context_conditions(context):
        raw = str(cond.get("raw") or cond.get("label") or "")
        chips.append(
            f"<span class='chip condition'><small>{direction_label(str(cond.get('direction')))}</small>"
            f"<b>{html.escape(raw)}</b></span>"
        )
    body = "".join(chips) if chips else "<span class='muted'>별도 조건 없음</span>"
    st.markdown(
        f"<div class='context'><div class='context-title'>◎ {html.escape(title)}</div>"
        f"<div class='chips'>{body}</div></div>",
        unsafe_allow_html=True,
    )


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
    return {
        "kind": kind,
        "target": target,
        "context": dict(context),
        "rows": rows,
        "hidden_rows": hidden_norm,
        "rfm": rfm_summary,
        "eda": eda,
        "rca": rca,
        "rashomon": rashomon,
        "wald": wald,
        "consensus": consensus,
        "conflict_insights": conflicts,
        "condition_evidence": condition_evidence,
        "decision": decision,
    }


def bar(label: str, value: float, text: str, tone: str = "gold") -> None:
    width = max(0, min(100, float(value)))
    st.markdown(
        f"<div class='bar'><div><span>{html.escape(label)}</span><b>{html.escape(text)}</b></div>"
        f"<i><em class='{tone}' style='width:{width:.1f}%'></em></i></div>",
        unsafe_allow_html=True,
    )


def pills(items: list[dict], tone: str) -> str:
    if not items:
        return "<span class='muted'>반복 키워드 없음</span>"
    return "".join(
        f"<span class='kw {tone}'>{html.escape(str(x.get('keyword','')))} "
        f"<b>{int(x.get('count',0))}</b></span>"
        for x in items[:5]
    )


def evidence_samples(title: str, samples: list[dict]) -> None:
    if not samples:
        return
    st.markdown(f"**{title}**")
    for x in samples[:2]:
        st.markdown(
            f"<div class='quote'><small>{html.escape(str(x.get('source') or 'Evidence'))}</small>"
            f"<b>{html.escape(str(x.get('title') or x.get('evidence_id') or ''))}</b>"
            f"<p>{html.escape(str(x.get('snippet') or ''))}</p></div>",
            unsafe_allow_html=True,
        )


st.markdown("""
<style>
:root{
  --ink:#252721;--sub:#6c6d65;--paper:#fffdf7;--gold:#c3a364;--gold2:#9b793e;
  --olive:#747b55;--navy:#1d3345;--navy2:#294b63;--line:#ddcfb7;--green:#668772;--red:#a96864
}
html,body,[class*="css"]{
  font-family:Pretendard,-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Segoe UI",sans-serif
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(circle at 18% 5%,rgba(195,163,100,.14),transparent 25rem),
    radial-gradient(circle at 85% 20%,rgba(116,123,85,.10),transparent 24rem),
    linear-gradient(180deg,#faf7f0 0%,#eee5d5 52%,#f8f3e9 100%);
  color:var(--ink)
}
[data-testid="stHeader"]{background:transparent}
.block-container{max-width:880px;padding-top:.75rem;padding-bottom:5rem;position:relative;z-index:2}
.agora-bg{
  position:fixed;z-index:0;left:0;right:0;top:58px;height:540px;pointer-events:none;
  opacity:.22;overflow:hidden
}
.agora-bg svg{width:100%;min-width:900px;height:100%;position:absolute;left:50%;transform:translateX(-50%)}
.agora-fade{
  position:fixed;z-index:1;left:0;right:0;top:58px;height:560px;pointer-events:none;
  background:linear-gradient(180deg,rgba(250,247,240,.18),rgba(250,247,240,.68) 64%,#f5efe5 96%)
}
.hero{
  position:relative;z-index:3;overflow:hidden;border-radius:28px;padding:2.3rem 2.15rem 1.8rem;
  color:#fffaf0;
  background:
    radial-gradient(circle at 50% -15%,rgba(232,207,151,.18),transparent 38%),
    linear-gradient(145deg,#183044 0%,#294b63 54%,#172d3d 100%);
  border:1px solid rgba(222,188,119,.78);
  box-shadow:0 22px 54px rgba(29,42,49,.22),inset 0 0 0 5px rgba(255,255,255,.025)
}
.hero:before{content:"";position:absolute;inset:8px;border:1px solid rgba(221,188,119,.42);border-radius:21px}
.hero:after{content:"";position:absolute;left:8%;right:8%;bottom:18px;height:1px;background:linear-gradient(90deg,transparent,#e0bf7c,transparent)}
.hero-copy{position:relative;z-index:2;max-width:640px}
.hero-mark{position:absolute;right:1rem;top:.8rem;width:165px;height:165px;opacity:.48}
.eyebrow{font-size:.69rem;font-weight:850;letter-spacing:.18em;color:#dfc38b}
.hero h1{font-size:2.35rem;line-height:1.25;letter-spacing:-.045em;margin:.82rem 0 .7rem;font-weight:850}
.hero h1 span{color:#f0d69d}
.hero p{font-size:.94rem;line-height:1.72;color:#edf0ec;margin:0;max-width:600px}
.hero-tags{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:1rem;padding-bottom:.42rem}
.hero-tags span{font-size:.69rem;padding:.32rem .62rem;border:1px solid rgba(232,205,150,.34);border-radius:999px;background:rgba(255,255,255,.055)}
[data-testid="stVerticalBlockBorderWrapper"]>div{
  background:rgba(255,253,247,.90);border-color:#ddcfb7!important;border-radius:20px!important;
  box-shadow:0 10px 28px rgba(72,58,35,.06);backdrop-filter:blur(4px)
}
.section-title{font-size:1.2rem;font-weight:850;letter-spacing:-.025em;margin:.45rem 0 .18rem}
.section-sub{font-size:.85rem;line-height:1.55;color:var(--sub);margin-bottom:.7rem}
.result,.condition-card,.balance,.verdict{
  background:rgba(255,253,247,.95);border:1px solid var(--line);border-radius:19px;padding:1rem;
  box-shadow:0 8px 24px rgba(68,56,37,.06);backdrop-filter:blur(4px)
}
.result{border-top:3px solid var(--gold)}.mini{font-size:.68rem;font-weight:850;letter-spacing:.13em;color:var(--gold2)}
.result h2{font-size:1.55rem;letter-spacing:-.035em;margin:.28rem 0}.result p{font-size:.9rem;line-height:1.6;color:#5c6058}
.context{background:rgba(253,248,238,.95);border:1px solid #d8c5a1;border-radius:16px;padding:.78rem;margin:.65rem 0}
.context-title{font-size:.76rem;font-weight:850;color:#71572f;margin-bottom:.45rem}.chips{display:flex;gap:.35rem;flex-wrap:wrap}
.chip{display:inline-flex;gap:.28rem;align-items:center;background:#fffdf8;border:1px solid #ded0b8;border-radius:999px;padding:.32rem .54rem;font-size:.72rem}
.chip small{color:#888982}.chip.condition{border-color:#c7ae77;background:#fff6e4}
.stat{background:rgba(255,253,248,.97);border:1px solid var(--line);border-radius:14px;padding:.7rem}
.stat small{color:var(--sub);font-weight:700}.stat strong{display:block;font-size:1.28rem}.stat em{font-size:.69rem;color:#85867e;font-style:normal}
.condition-card{margin:.6rem 0;border-left:3px solid var(--gold)}
.condition-head{display:flex;justify-content:space-between;gap:.5rem}.condition-head strong{font-size:1rem}
.badge{font-size:.69rem;color:#6d5430;background:#f2e6d2;border-radius:999px;padding:.25rem .48rem;white-space:nowrap}
.condition-note{font-size:.77rem;color:#70726b;margin-top:.25rem}
.bar{margin:.48rem 0}.bar>div{display:flex;justify-content:space-between;font-size:.76rem;margin-bottom:.2rem;color:#60635c}
.bar i{display:block;height:8px;border-radius:99px;background:#eae3d7;overflow:hidden}.bar em{display:block;height:100%;border-radius:99px}
.bar em.positive{background:var(--green)}.bar em.negative{background:var(--red)}.bar em.gold{background:var(--gold)}
.kwrow{display:flex;flex-wrap:wrap;gap:.3rem;margin:.35rem 0 .65rem}.kw{font-size:.7rem;border-radius:999px;padding:.27rem .48rem;border:1px solid}
.kw.positive{background:#eef5ef;border-color:#c6d7c8;color:#49634f}.kw.negative{background:#f8eeee;border-color:#e4c7c6;color:#7d4e4a}
.muted{font-size:.73rem;color:#8b8c85}.quote{border-left:3px solid #b8975d;background:#faf5ea;border-radius:0 10px 10px 0;padding:.58rem .66rem;margin:.38rem 0}
.quote small{display:block;color:#8a7b63;font-weight:800;font-size:.63rem}.quote b{display:block;font-size:.78rem;margin:.1rem 0}.quote p{font-size:.74rem;line-height:1.45;color:#61635d;margin:0}
.signals{display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem}.signal{background:#f6f0e5;border:1px solid #d8cab3;border-radius:13px;padding:.65rem}
.signal small{color:var(--sub);display:block}.signal b{font-size:1.05rem}
.balance{border-top:3px solid var(--gold)}.balance-grid{display:grid;grid-template-columns:1fr 52px 1fr;align-items:center;gap:.5rem;margin-top:.6rem}
.side{text-align:center;background:#faf5e9;border:1px solid #ddcfb5;border-radius:14px;padding:.72rem}.side small{color:#75766e}.score{font-size:1.6rem;font-weight:900}.scale{text-align:center;font-size:1.45rem;color:#a08049}
.verdict{margin-top:.8rem;border:2px solid #c4a05e;box-shadow:inset 0 0 0 4px #f8efdd,0 12px 30px rgba(83,61,27,.10)}
.verdict-title{font-size:.75rem;font-weight:900;letter-spacing:.08em;color:#74552a;margin-bottom:.45rem}
[data-testid="stMetric"]{background:#fffdf8;border:1px solid var(--line);padding:.52rem .58rem;border-radius:12px}
[data-testid="stMetricValue"]{font-size:1.15rem;font-weight:850}.stButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:999px;min-height:2.75rem;font-weight:750}
.stTextInput input{border-radius:13px!important;background:#fffdf8!important}[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:rgba(255,253,248,.90)}
@media(max-width:700px){
  .block-container{padding:.5rem .72rem 4.5rem}
  .agora-bg{top:68px;height:430px;opacity:.17}.agora-bg svg{min-width:760px}
  .agora-fade{top:68px;height:450px;background:linear-gradient(180deg,rgba(250,247,240,.12),rgba(250,247,240,.72) 70%,#f5efe5 100%)}
  .hero{border-radius:22px;padding:1.45rem 1.05rem 1.15rem}
  .hero:before{inset:6px;border-radius:17px}.hero-mark{width:105px;height:105px;right:-.2rem;top:.25rem;opacity:.42}
  .eyebrow{font-size:.56rem;letter-spacing:.12em}.hero h1{font-size:1.62rem;line-height:1.3;max-width:94%;margin:.62rem 0 .5rem}
  .hero p{font-size:.79rem;line-height:1.58;max-width:96%}.hero-tags{gap:.26rem;margin-top:.7rem}.hero-tags span{font-size:.6rem;padding:.25rem .42rem}
  .section-title{font-size:1.07rem}.section-sub{font-size:.78rem}
  .result,.condition-card,.balance,.verdict{border-radius:15px;padding:.8rem}.result h2{font-size:1.33rem}.result p{font-size:.81rem}
  .chips{gap:.26rem}.chip{font-size:.66rem;padding:.27rem .43rem}.condition-head{display:block}.badge{display:inline-block;margin-top:.28rem}
  .signals{grid-template-columns:1fr}.balance-grid{grid-template-columns:1fr 36px 1fr;gap:.28rem}.side{padding:.55rem .25rem}.score{font-size:1.33rem}.scale{font-size:1.1rem}
  [data-testid="column"]{min-width:0!important}
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='agora-bg' aria-hidden='true'>
<svg viewBox='0 0 1200 520'>
  <g fill='none' stroke='#88775d' stroke-width='3.2'>
    <path d='M50 430H1150M110 395H1090M180 365H1020'/>
    <ellipse cx='600' cy='444' rx='440' ry='65'/>
    <path d='M330 365V170M390 365V170M810 365V170M870 365V170'/>
    <path d='M305 170h110l-55-58zM785 170h110l-55-58z'/>
    <path d='M450 365V130M510 365V130M570 365V130M630 365V130M690 365V130M750 365V130'/>
    <path d='M420 130h360L600 42zM410 365h380'/>
    <path d='M235 400c75-45 115-60 170-65M965 400c-75-45-115-60-170-65'/>
  </g>
</svg>
</div>
<div class='agora-fade'></div>
<div class='hero'>
  <div class='hero-copy'>
    <div class='eyebrow'>DIKE'S EYE · CONDITIONAL JUDGMENT</div>
    <h1>평균의 진실이 아니라,<br><span>당신에게 유효한 진실을 판별합니다.</span></h1>
    <p>아고라의 수많은 목소리처럼 리뷰도 서로 다릅니다. 디케는 전체 여론, 당신의 조건, 엇갈린 증거와 리뷰 밖의 신호를 하나의 저울에 올려 이번 선택에 유효한 진실을 판별합니다.</p>
    <div class='hero-tags'><span>CONSENSUS</span><span>CONDITION</span><span>RASHOMON</span><span>WALD</span><span>JUDGMENT</span></div>
  </div>
  <svg class='hero-mark' viewBox='0 0 180 180' aria-hidden='true'>
    <g fill='none' stroke='#e2c17f' stroke-width='2.3'><circle cx='90' cy='88' r='55'/><path d='M90 42v74M56 68h68M64 68L47 98h34L64 68zm52 0L99 98h34l-17-30zM70 120h40'/></g>
    <g fill='#7e8960'><ellipse cx='39' cy='122' rx='5' ry='12' transform='rotate(-35 39 122)'/><ellipse cx='31' cy='103' rx='5' ry='12' transform='rotate(-55 31 103)'/><ellipse cx='32' cy='82' rx='5' ry='12' transform='rotate(-75 32 82)'/><ellipse cx='141' cy='122' rx='5' ry='12' transform='rotate(35 141 122)'/><ellipse cx='149' cy='103' rx='5' ry='12' transform='rotate(55 149 103)'/><ellipse cx='148' cy='82' rx='5' ry='12' transform='rotate(75 148 82)'/></g>
  </svg>
</div>
""", unsafe_allow_html=True)

DEFAULTS = {
    "intent": None, "parsed_context": {}, "candidates": [], "selected_target": None,
    "analysis": None, "user_report": None, "explanation": None, "last_error": ""
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.container(border=True):
    st.markdown("<div class='section-title'>어떤 선택을 고민하고 있나요?</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>대상과 상황, 중요하게 보는 조건을 평소 말하듯 적어주세요. 허용하거나 피하고 싶은 조건도 함께 읽습니다.</div>", unsafe_allow_html=True)
    st.caption("식당 예시 · 토요일 7시 데이트인데 성수 어니언 어때? 분위기 좋고 조용한 게 중요해")
    st.caption("상품 예시 · 출퇴근용으로 소니 WH-1000XM6 어때? 착용감과 배터리가 중요하고 무거운 건 싫어")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("질문", placeholder="대상 + 내 상황 + 중요조건을 자유롭게 입력하세요", label_visibility="collapsed")
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
                st.session_state.candidates = candidates or [{
                    "title": intent.get("target") or intent.get("original", ""),
                    "category": "직접 입력", "address": ""
                }]
            except Exception as exc:
                st.session_state.last_error = f"{type(exc).__name__}: {exc}"
        if st.session_state.get("candidates"):
            labels = [f"{x.get('title','')} · {x.get('category','')} · {x.get('address','')}" for x in st.session_state.candidates]
            idx = st.radio("이 장소가 맞나요?", range(len(labels)), format_func=lambda i: labels[i])
            if st.button("이 장소가 맞아요", type="primary", use_container_width=True):
                chosen = st.session_state.candidates[idx]
                st.session_state.selected_target = {
                    "kind": "restaurant",
                    "name": chosen.get("title") or intent.get("target", ""),
                    "meta": chosen,
                }

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
            st.session_state.analysis = analysis
            st.session_state.user_report = report
        except Exception as exc:
            st.session_state.last_error = f"{type(exc).__name__}: {exc}"

if st.session_state.analysis and st.session_state.user_report:
    a = st.session_state.analysis
    r = st.session_state.user_report
    d = a["decision"]
    comps = d.get("components", {})
    rows = a.get("rows", [])
    hidden = a.get("hidden_rows", [])
    conditions = a.get("rca", {}).get("condition_results", [])
    consensus = a.get("consensus", {})
    conflicts = a.get("conflict_insights", [])
    details = {str(x.get("aspect")): x for x in a.get("condition_evidence", [])}

    st.markdown(
        f"<div class='result'><div class='mini'>DIKE'S JUDGMENT</div>"
        f"<h2>{html.escape(str(r.get('headline','')))}</h2>"
        f"<p>{html.escape(str(r.get('summary','')))}</p></div>",
        unsafe_allow_html=True,
    )
    render_context_chips(a.get("context", {}), "이번 판정 조건")

    c1, c2, c3 = st.columns(3)
    for col, label, value, note in [
        (c1, "최종 적합도", f"{d.get('fit_score',0):.0f}/100", "전체+조건"),
        (c2, "판단 신뢰도", f"{d.get('confidence',0):.0f}%", "근거 품질"),
        (c3, "Evidence", f"{len(rows)+len(hidden)}건", "검토 표본"),
    ]:
        col.markdown(
            f"<div class='stat'><small>{label}</small><strong>{value}</strong><em>{note}</em></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Ⅰ. 시민의 목소리 · 전체 여론</div><div class='section-sub'>아고라의 전체 목소리처럼, 일반 후기 중심으로 대체적인 방향부터 확인합니다.</div>", unsafe_allow_html=True)
    total = int(consensus.get("sample_count", 0))
    pos = int(consensus.get("positive_count", 0))
    neg = int(consensus.get("negative_count", 0))
    pr = float(consensus.get("positive_rate", 0)) * 100
    nr = float(consensus.get("negative_rate", 0)) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("긍정", f"{pr:.0f}%", f"{pos}/{total}건")
    c2.metric("부정", f"{nr:.0f}%", f"{neg}/{total}건")
    c3.metric("여론 점수", f"{float(consensus.get('opinion_score',50)):.0f}/100")
    bar("일반 후기 긍정", pr, f"{pos}건 · {pr:.0f}%", "positive")
    st.markdown(f"<div class='kwrow'>{pills(consensus.get('positive_keywords',[]),'positive')}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='kwrow'>{pills(consensus.get('negative_keywords',[]),'negative')}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ⅱ. 당신의 조건</div><div class='section-sub'>평균과 별개로, 당신이 실제로 중요하게 말한 조건을 하나씩 따로 판정합니다.</div>", unsafe_allow_html=True)
    if conditions:
        for item in conditions:
            aspect = str(item.get("aspect") or "")
            detail = details.get(aspect, {})
            label = str(item.get("label") or aspect_label(aspect))
            total = int(item.get("total_count", 0))
            pos = int(item.get("positive_count", 0))
            neg = int(item.get("negative_count", 0))
            pr = float(item.get("positive_rate", 0)) * 100
            fit = float(item.get("fit", .5)) * 100
            st.markdown(
                f"<div class='condition-card'><div class='condition-head'><div><strong>◎ {html.escape(label)}</strong>"
                f"<div class='condition-note'>{html.escape(str(item.get('raw') or label))} · 관련 {total}건 · 직접 근거 {int(item.get('direct_count',0))}건</div>"
                f"</div><span class='badge'>{direction_label(str(item.get('direction')))} · 중요도 {float(item.get('importance',.8))*100:.0f}%</span></div></div>",
                unsafe_allow_html=True,
            )
            if total >= 3:
                c1, c2, c3 = st.columns(3)
                c1.metric("긍정", f"{pr:.0f}%", f"{pos}/{total}건")
                c2.metric("부정", f"{100-pr:.0f}%", f"{neg}/{total}건")
                c3.metric("조건 적합", f"{fit:.0f}/100")
                bar("긍정 Evidence", pr, f"{pos}건", "positive")
                st.markdown(f"<div class='kwrow'>{pills(detail.get('positive_keywords',[]),'positive')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kwrow'>{pills(detail.get('negative_keywords',[]),'negative')}</div>", unsafe_allow_html=True)
                if int(item.get("situational_count", 0)) >= 3:
                    st.info(
                        f"내 상황과 직접 맞는 {int(item.get('situational_count',0))}건에서는 "
                        f"부정 {float(item.get('situational_negative_rate',0))*100:.0f}% · "
                        f"전체 대비 {float(item.get('situational_lift',0))*100:+.1f}%p"
                    )
                with st.expander(f"{label} 실제 근거"):
                    evidence_samples("긍정 Evidence", detail.get("positive_samples", []))
                    evidence_samples("부정 Evidence", detail.get("negative_samples", []))
            else:
                st.warning(f"{label} 관련 Evidence가 {total}건이라 판단 안정성이 낮습니다.")

    if conflicts:
        st.markdown("<div class='section-title'>Ⅲ. Rashomon · 갈라진 진실</div><div class='section-sub'>같은 대상에 대한 서로 다른 진실을 양쪽 Evidence로 나눠 봅니다.</div>", unsafe_allow_html=True)
        for x in conflicts[:3]:
            with st.container(border=True):
                pc = int(x.get("positive_count", 0))
                nc = int(x.get("negative_count", 0))
                t = max(1, pc + nc)
                st.markdown(f"**{html.escape(str(x.get('label') or '쟁점'))}** · 긍정 {pc}건 vs 부정 {nc}건")
                bar("긍정", pc / t * 100, f"{pc}건", "positive")
                bar("부정", nc / t * 100, f"{nc}건", "negative")
                st.markdown(f"<div class='kwrow'>{pills(x.get('positive_keywords',[]),'positive')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kwrow'>{pills(x.get('negative_keywords',[]),'negative')}</div>", unsafe_allow_html=True)
                with st.expander("양쪽 실제 Evidence"):
                    evidence_samples("긍정 쪽", x.get("positive_samples", []))
                    evidence_samples("부정 쪽", x.get("negative_samples", []))

    signals = a.get("wald", {}).get("signal_counts", {})
    if signals:
        st.markdown("<div class='section-title'>Ⅳ. Wald · 사라진 진실</div><div class='section-sub'>일반 리뷰에 잘 남지 않는 이탈·실패·불편 신호를 별도로 확인합니다.</div>", unsafe_allow_html=True)
        top = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:3]
        cards = "".join(
            f"<div class='signal'><small>{html.escape(WALD_LABELS.get(k,k))}</small><b>{v}건</b></div>"
            for k, v in top
        )
        st.markdown(f"<div class='signals'>{cards}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ⅴ. Dike의 저울</div>", unsafe_allow_html=True)
    overall = float(comps.get("consensus_score", 50))
    personal = float(comps.get("condition_score", 50))
    delta = personal - overall
    st.markdown(
        f"<div class='balance'><div class='mini'>BALANCE OF TRUTH</div><div class='balance-grid'>"
        f"<div class='side'><small>전체 여론</small><div class='score'>{overall:.0f}</div></div>"
        f"<div class='scale'>⚖</div>"
        f"<div class='side'><small>나에게 유효한 진실</small><div class='score'>{personal:.0f}</div></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    if abs(delta) < 7:
        st.caption("전체 여론과 내 조건이 대체로 같은 방향입니다.")
    elif delta > 0:
        st.caption(f"평균보다 내 조건에서 {delta:.0f}점 더 유리합니다. 대중적 평가보다 나에게 더 잘 맞을 수 있습니다.")
    else:
        st.caption(f"평균보다 내 조건에서 {abs(delta):.0f}점 불리합니다. 대체로 좋은 평가여도 이번 선택에는 조건이 붙습니다.")

    st.markdown("<div class='verdict'><div class='verdict-title'>Ⅵ. DIKE'S JUDGMENT · SOLOMON CHOICE</div>", unsafe_allow_html=True)
    for item in r.get("recommendations", []):
        st.markdown(f"- **{item}**")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("AI가 판정 이유를 더 자세히 설명"):
        if st.button("설명 생성", use_container_width=True):
            with st.spinner("판정 근거를 정리하고 있어요..."):
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
        st.json(consensus)
        df = pd.DataFrame(conditions)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.json({"consensus_score": overall, "condition_score": personal, "policy": d.get("policy", {})})

    if st.button("새로운 질문하기", use_container_width=True):
        for key in list(DEFAULTS) + ["ctx_day", "ctx_time", "ctx_purpose", "ctx_preference", "target_edit"]:
            st.session_state.pop(key, None)
        st.rerun()

if st.session_state.last_error:
    st.error("처리 중 문제가 생겼어요. 입력 내용은 유지되어 있습니다.")
    with st.expander("오류 정보"):
        st.code(st.session_state.last_error)

st.divider()
st.caption("Dike's Eye · Consensus × Condition × Rashomon × Wald → Judgment")

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

RAW = "https://raw.githubusercontent.com/dooyoung94/dikes-eye-poc/main/assets/dikes_eye_asset_pack/assets"
BG_DESKTOP = f"{RAW}/background/agora_bg_desktop.webp"
BG_MOBILE = f"{RAW}/background/agora_bg_center.webp"
HERO_FRAME = f"{RAW}/hero/hero_frame.svg"
HERO_SCALE = f"{RAW}/hero/scales_gold.svg"
HERO_LAUREL = f"{RAW}/hero/laurel_gold.svg"
DIVIDER = f"{RAW}/hero/divider_gold.svg"
PARCHMENT = f"{RAW}/cards/parchment_card_texture.webp"
INPUT_TEXTURE = f"{RAW}/cards/input_box_texture.webp"
CARD_CORNER = f"{RAW}/cards/card_corner.svg"
MARBLE = f"{RAW}/texture/marble_texture.webp"
ICON = {
    "consensus": f"{RAW}/icons/consensus_icon.svg",
    "condition": f"{RAW}/icons/condition_icon.svg",
    "rashomon": f"{RAW}/icons/rashomon_icon.svg",
    "wald": f"{RAW}/icons/wald_icon.svg",
    "balance": f"{RAW}/icons/balance_icon.svg",
    "solomon": f"{RAW}/icons/solomon_icon.svg",
}


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


st.markdown(f"""
<style>
:root{{--ink:#1a1a1a;--sub:#686868;--paper:#fff3ea;--gold:#d4af37;--champ:#f2d99c;--navy:#0f1e2d;--navy2:#142739;--line:#d8c39a;--green:#6c8c70;--red:#a75f5b}}
html,body,[class*="css"]{{font-family:Pretendard,-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Segoe UI",sans-serif}}
[data-testid="stAppViewContainer"]{{color:var(--ink);background:url('{BG_DESKTOP}') center top/cover fixed no-repeat;background-color:#e9ddc9}}
[data-testid="stAppViewContainer"]:before{{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;background:linear-gradient(180deg,rgba(255,248,237,.03),rgba(248,239,224,.20) 64%,rgba(244,235,218,.36))}}
[data-testid="stHeader"]{{background:rgba(250,246,238,.74);backdrop-filter:blur(9px)}}
.block-container{{max-width:1120px;padding-top:.8rem;padding-bottom:5rem;position:relative;z-index:2}}
.hero{{position:relative;overflow:hidden;border-radius:32px;padding:3.2rem 3.2rem 2.3rem;text-align:center;color:#fff7e7;background:linear-gradient(145deg,#0b1a28,#153148 55%,#0b1a28);box-shadow:0 26px 60px rgba(24,30,34,.30);isolation:isolate}}
.hero:before{{content:"";position:absolute;inset:0;background:url('{HERO_FRAME}') center/100% 100% no-repeat;z-index:0;pointer-events:none}}
.hero-laurel{{position:absolute;inset:13% 10% auto 10%;height:58%;background:url('{HERO_LAUREL}') center/contain no-repeat;opacity:.86;z-index:0;pointer-events:none}}
.hero-scale{{width:74px;height:74px;object-fit:contain;margin:0 auto .45rem;filter:drop-shadow(0 3px 7px rgba(0,0,0,.22))}}
.hero-copy{{position:relative;z-index:2;max-width:760px;margin:auto}}
.eyebrow{{font-size:.72rem;font-weight:850;letter-spacing:.18em;color:#ecd094}}
.hero h1{{font-size:2.5rem;line-height:1.28;letter-spacing:-.045em;margin:.8rem 0 .7rem;font-weight:850;color:#f3d594;text-shadow:0 2px 5px rgba(0,0,0,.22)}}
.hero p{{font-size:.94rem;line-height:1.8;color:#f5f2e9;margin:.2rem auto 0;max-width:680px}}
.hero-divider{{width:min(480px,70%);height:30px;background:url('{DIVIDER}') center/contain no-repeat;margin:.45rem auto .2rem}}
.hero-tags{{display:flex;justify-content:center;gap:.42rem;flex-wrap:wrap;margin-top:.7rem}}.hero-tags span{{font-size:.68rem;padding:.3rem .58rem;color:#ead5a5;border-right:1px solid rgba(242,217,156,.35)}}.hero-tags span:last-child{{border-right:0}}
.intro-title{{text-align:center;font-size:1rem;font-weight:850;margin:1.05rem 0 .65rem;color:#574321;text-shadow:0 1px 0 #fff}}
.journey-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:.55rem;margin-bottom:1rem}}
.journey-card{{position:relative;min-height:190px;padding:1rem .65rem .85rem;text-align:center;background:linear-gradient(rgba(255,250,239,.94),rgba(255,250,239,.94)),url('{PARCHMENT}') center/cover;border:1px solid rgba(188,145,67,.48);border-radius:15px;box-shadow:0 10px 24px rgba(70,49,25,.10);overflow:hidden}}
.journey-card:after{{content:"";position:absolute;right:0;bottom:0;width:54px;height:54px;background:url('{CARD_CORNER}') right bottom/contain no-repeat;opacity:.62}}
.journey-card img{{width:58px;height:58px;object-fit:contain;margin:.1rem auto .45rem;display:block;filter:drop-shadow(0 4px 8px rgba(45,36,20,.17))}}
.journey-card b{{display:block;font-size:.82rem;line-height:1.35;color:#1c2630}}.journey-card p{{font-size:.68rem;line-height:1.48;color:#665f55;margin:.4rem 0 0}}
[data-testid="stVerticalBlockBorderWrapper"]>div{{background:linear-gradient(rgba(255,252,244,.97),rgba(255,252,244,.97)),url('{PARCHMENT}') center/cover!important;border-color:#d7bd8b!important;border-radius:20px!important;box-shadow:0 12px 30px rgba(70,50,28,.12)!important}}
.section-title{{font-size:1.2rem;font-weight:850;letter-spacing:-.025em;margin:.55rem 0 .18rem;color:#241d15}}.section-sub{{font-size:.84rem;line-height:1.55;color:#5f5a52;margin-bottom:.7rem}}
.result,.condition-card,.balance,.verdict{{background:linear-gradient(rgba(255,252,244,.98),rgba(255,252,244,.98)),url('{PARCHMENT}') center/cover;border:1px solid #d3b77e;border-radius:18px;padding:1rem;box-shadow:0 10px 25px rgba(68,49,28,.11)}}
.result{{border-top:3px solid var(--gold)}}.mini{{font-size:.68rem;font-weight:850;letter-spacing:.13em;color:#8b672d}}.result h2{{font-size:1.55rem;letter-spacing:-.035em;margin:.28rem 0}}.result p{{font-size:.9rem;line-height:1.6;color:#555047}}
.context{{background:linear-gradient(rgba(255,250,239,.97),rgba(255,250,239,.97)),url('{INPUT_TEXTURE}') center/cover;border:1px solid #d5b97f;border-radius:16px;padding:.78rem;margin:.65rem 0;box-shadow:0 6px 18px rgba(92,70,37,.07)}}
.context-title{{font-size:.76rem;font-weight:850;color:#71572f;margin-bottom:.45rem}}.chips{{display:flex;gap:.35rem;flex-wrap:wrap}}.chip{{display:inline-flex;gap:.28rem;align-items:center;background:#fffdf8;border:1px solid #decba5;border-radius:999px;padding:.32rem .54rem;font-size:.72rem}}.chip small{{color:#777970}}.chip.condition{{border-color:#c9a85e;background:#fff2d7}}
.stat{{background:#fffaf1;border:1px solid #dcc79f;border-radius:14px;padding:.7rem;box-shadow:0 5px 15px rgba(70,55,35,.06)}}.stat small{{color:#62645e;font-weight:700}}.stat strong{{display:block;font-size:1.28rem}}.stat em{{font-size:.69rem;color:#777a72;font-style:normal}}
.condition-card{{margin:.6rem 0;border-left:3px solid var(--gold)}}.condition-head{{display:flex;justify-content:space-between;gap:.5rem}}.condition-head strong{{font-size:1rem}}.badge{{font-size:.69rem;color:#6d5430;background:#f2e6d2;border-radius:999px;padding:.25rem .48rem;white-space:nowrap}}.condition-note{{font-size:.77rem;color:#656861;margin-top:.25rem}}
.bar{{margin:.48rem 0}}.bar>div{{display:flex;justify-content:space-between;font-size:.76rem;margin-bottom:.2rem;color:#5d6059}}.bar i{{display:block;height:8px;border-radius:99px;background:#e9e1d3;overflow:hidden}}.bar em{{display:block;height:100%;border-radius:99px}}.bar em.positive{{background:var(--green)}}.bar em.negative{{background:var(--red)}}.bar em.gold{{background:var(--gold)}}
.kwrow{{display:flex;flex-wrap:wrap;gap:.3rem;margin:.35rem 0 .65rem}}.kw{{font-size:.7rem;border-radius:999px;padding:.27rem .48rem;border:1px solid}}.kw.positive{{background:#eef5ef;border-color:#c6d7c8;color:#49634f}}.kw.negative{{background:#f8eeee;border-color:#e4c7c6;color:#7d4e4a}}.muted{{font-size:.73rem;color:#777a72}}
.quote{{border-left:3px solid #b8975d;background:#faf4e7;border-radius:0 10px 10px 0;padding:.58rem .66rem;margin:.38rem 0}}.quote small{{display:block;color:#806f58;font-weight:800;font-size:.63rem}}.quote b{{display:block;font-size:.78rem;margin:.1rem 0}}.quote p{{font-size:.74rem;line-height:1.45;color:#555850;margin:0}}
.signals{{display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem}}.signal{{background:#f7f0e4;border:1px solid #d5c5aa;border-radius:13px;padding:.65rem}}.signal small{{color:#63665f;display:block}}.signal b{{font-size:1.05rem}}
.balance{{border-top:3px solid var(--gold)}}.balance-grid{{display:grid;grid-template-columns:1fr 52px 1fr;align-items:center;gap:.5rem;margin-top:.6rem}}.side{{text-align:center;background:#faf4e7;border:1px solid #d9c7a6;border-radius:14px;padding:.72rem}}.side small{{color:#676961}}.score{{font-size:1.6rem;font-weight:900}}.scale{{text-align:center;font-size:1.45rem;color:#a08049}}
.verdict{{margin-top:.8rem;border:2px solid #c4a05e;box-shadow:inset 0 0 0 4px #f8efdd,0 12px 30px rgba(83,61,27,.14)}}.verdict-title{{font-size:.75rem;font-weight:900;letter-spacing:.08em;color:#74552a;margin-bottom:.45rem}}
[data-testid="stMetric"]{{background:#fffaf1;border:1px solid #dcc79f;padding:.52rem .58rem;border-radius:12px}}[data-testid="stMetricValue"]{{font-size:1.15rem;font-weight:850}}.stButton>button,[data-testid="stFormSubmitButton"]>button{{border-radius:12px;min-height:2.8rem;font-weight:800;background:linear-gradient(135deg,#11283c,#173b55)!important;color:#f0d397!important;border:1px solid #d4af37!important;box-shadow:0 7px 18px rgba(22,40,55,.18)}}.stTextInput input{{border-radius:12px!important;background:#fffaf1!important;color:#272923!important}}[data-testid="stExpander"]{{border:1px solid #d5c09a;border-radius:14px;background:#fffaf2}}
@media(max-width:700px){{[data-testid="stAppViewContainer"]{{background:url('{BG_MOBILE}') center top/auto 100vh fixed no-repeat;background-color:#e7d8c0}}.block-container{{padding:.5rem .65rem 4rem}}.hero{{border-radius:20px;padding:1.5rem .9rem 1.18rem}}.hero-laurel{{inset:10% 2% auto 2%;height:57%;opacity:.55}}.hero-scale{{width:52px;height:52px}}.eyebrow{{font-size:.54rem;letter-spacing:.11em}}.hero h1{{font-size:1.55rem;line-height:1.31;margin:.48rem 0}}.hero p{{font-size:.76rem;line-height:1.6}}.hero-tags{{gap:.1rem;margin-top:.45rem}}.hero-tags span{{font-size:.55rem;padding:.2rem .32rem}}.intro-title{{font-size:.9rem}}.journey-grid{{grid-template-columns:repeat(2,1fr);gap:.42rem}}.journey-card{{min-height:155px;padding:.72rem .4rem}}.journey-card img{{width:48px;height:48px}}.journey-card b{{font-size:.73rem}}.journey-card p{{font-size:.62rem}}.section-title{{font-size:1.05rem}}.section-sub{{font-size:.77rem}}.result,.condition-card,.balance,.verdict{{padding:.8rem;border-radius:15px}}.condition-head{{display:block}}.badge{{display:inline-block;margin-top:.28rem}}.signals{{grid-template-columns:1fr}}.balance-grid{{grid-template-columns:1fr 36px 1fr;gap:.28rem}}[data-testid="column"]{{min-width:0!important}}}}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class='hero'>
 <div class='hero-laurel'></div>
 <div class='hero-copy'>
  <div class='eyebrow'>DIKE'S EYE · CONDITIONAL JUDGMENT AGENT</div>
  <img class='hero-scale' src='{HERO_SCALE}' alt='Dike scale'>
  <h1>평균의 진실이 아니라,<br>당신에게 유효한 진실을 판별합니다.</h1>
  <div class='hero-divider'></div>
  <p>아고라의 수많은 목소리처럼 리뷰도 서로 다릅니다.<br>디케는 전체 여론, 당신의 조건, 엇갈린 증거와 리뷰 밖의 신호를 하나의 저울에 올려 이번 선택에 유효한 진실을 판별합니다.</p>
  <div class='hero-tags'><span>CONSENSUS</span><span>CONDITION</span><span>RASHOMON</span><span>WALD</span><span>JUDGMENT</span></div>
 </div>
</div>
<div class='intro-title'>Dike가 진실을 판별하는 6단계</div>
<div class='journey-grid'>
 <div class='journey-card'><img src='{ICON['consensus']}'><b>Ⅰ 시민의 목소리</b><p>일반 후기에서 전체 여론의 방향과 반복 키워드를 먼저 확인합니다.</p></div>
 <div class='journey-card'><img src='{ICON['condition']}'><b>Ⅱ 당신의 조건</b><p>사용자가 중요하게 말한 조건과 허용·회피 기준을 별도로 평가합니다.</p></div>
 <div class='journey-card'><img src='{ICON['rashomon']}'><b>Ⅲ Rashomon</b><p>같은 경험에서도 서로 엇갈린 긍정·부정 Evidence를 비교합니다.</p></div>
 <div class='journey-card'><img src='{ICON['wald']}'><b>Ⅳ Wald</b><p>리뷰에 남지 않거나 묻힌 실패·이탈·불편 신호를 추가 확인합니다.</p></div>
 <div class='journey-card'><img src='{ICON['balance']}'><b>Ⅴ Dike의 저울</b><p>전체 여론과 내 조건을 한 저울에 올려 차이를 정량 비교합니다.</p></div>
 <div class='journey-card'><img src='{ICON['solomon']}'><b>Ⅵ Solomon Choice</b><p>조건·근거·숨은 위험을 종합해 최종 선택과 확인사항을 제시합니다.</p></div>
</div>
""", unsafe_allow_html=True)

DEFAULTS = {"intent": None, "parsed_context": {}, "candidates": [], "selected_target": None, "analysis": None, "user_report": None, "explanation": None, "last_error": ""}
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
        submitted = st.form_submit_button("⚖ Dike에게 판단 맡기기", type="primary", use_container_width=True)

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

    st.markdown("<div class='section-title'>Ⅰ. 시민의 목소리 · 전체 여론</div><div class='section-sub'>일반 후기 중심으로 대체적인 방향부터 확인합니다.</div>", unsafe_allow_html=True)
    total = int(consensus.get("sample_count",0)); pos = int(consensus.get("positive_count",0)); neg = int(consensus.get("negative_count",0)); pr = float(consensus.get("positive_rate",0))*100; nr = float(consensus.get("negative_rate",0))*100
    c1,c2,c3 = st.columns(3); c1.metric("긍정",f"{pr:.0f}%",f"{pos}/{total}건"); c2.metric("부정",f"{nr:.0f}%",f"{neg}/{total}건"); c3.metric("여론 점수",f"{float(consensus.get('opinion_score',50)):.0f}/100")
    bar("일반 후기 긍정",pr,f"{pos}건 · {pr:.0f}%","positive")
    st.markdown(f"<div class='kwrow'>{pills(consensus.get('positive_keywords',[]),'positive')}</div>",unsafe_allow_html=True)
    st.markdown(f"<div class='kwrow'>{pills(consensus.get('negative_keywords',[]),'negative')}</div>",unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Ⅱ. 당신의 조건</div><div class='section-sub'>평균과 별개로, 실제로 중요하게 말한 조건을 하나씩 판정합니다.</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='section-title'>Ⅳ. Wald · 사라진 진실</div><div class='section-sub'>리뷰에 잘 남지 않는 이탈·실패·불편 신호를 따로 확인합니다.</div>",unsafe_allow_html=True)
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

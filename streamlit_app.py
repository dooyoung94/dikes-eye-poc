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


def intent_context(intent: dict | None) -> dict:
    intent = intent or {}
    return {
        "date_or_day": str(intent.get("date_or_day") or "").strip(),
        "time": str(intent.get("time") or "").strip(),
        "purpose": str(intent.get("purpose") or "").strip(),
        "preference": str(intent.get("preference") or "").strip(),
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
    parsed = st.session_state.get("parsed_context") or intent_context(
        st.session_state.get("intent")
    )
    values = {
        "date_or_day": str(st.session_state.get("ctx_day") or "").strip(),
        "time": str(st.session_state.get("ctx_time") or "").strip(),
        "purpose": str(st.session_state.get("ctx_purpose") or "").strip(),
        "preference": str(st.session_state.get("ctx_preference") or "").strip(),
    }
    # 위젯 상태가 비어도 질문에서 최초 파싱한 조건은 잃지 않는다.
    for key in values:
        if not values[key]:
            values[key] = str(parsed.get(key) or "").strip()
    return values


def context_items(context: dict, kind: str) -> list[tuple[str, str]]:
    if kind == "restaurant":
        candidates = [
            ("요일", context.get("date_or_day", "")),
            ("시간", context.get("time", "")),
            ("목적", context.get("purpose", "")),
            ("중요조건", context.get("preference", "")),
        ]
    else:
        candidates = [
            ("사용상황", context.get("date_or_day", "")),
            ("추가조건", context.get("time", "")),
            ("목적", context.get("purpose", "")),
            ("중요조건", context.get("preference", "")),
        ]
    return [
        (label, str(value).strip())
        for label, value in candidates
        if str(value).strip()
    ]


def render_context_chips(
    context: dict,
    kind: str,
    title: str = "내 조건",
) -> None:
    items = context_items(context, kind)
    if not items:
        st.markdown(
            f"<div class='context-panel'><div class='context-title'>🎯 {title}</div>"
            "<div class='context-empty'>질문에서 별도 조건을 찾지 못했어요. "
            "조건을 추가하면 그 조건을 포함해 다시 검색합니다.</div></div>",
            unsafe_allow_html=True,
        )
        return

    chips = "".join(
        f"<span class='context-chip'><span>{label}</span><strong>{value}</strong></span>"
        for label, value in items
    )
    st.markdown(
        f"<div class='context-panel'><div class='context-title'>🎯 {title}</div>"
        f"<div class='context-chips'>{chips}</div></div>",
        unsafe_allow_html=True,
    )


def run_decision(target: str, kind: str, context: dict) -> dict:
    creds = naver_credentials()
    visible_raw = collect_visible_evidence(
        target,
        kind=kind,
        context=context,
        **creds,
    )
    hidden_raw = collect_hidden_evidence(
        target,
        kind=kind,
        context=context,
        **creds,
    )
    visible_norm = normalize_evidence(visible_raw, context)
    hidden_norm = normalize_evidence(hidden_raw, context)
    rfm_rows, rfm_summary = build_rfm(visible_norm)
    eda = build_eda(rfm_rows)
    rca = derive_rca(rfm_rows, context)
    rashomon = build_rashomon(rca)
    wald = analyze_wald(hidden_norm, kind=kind)
    decision = score_decision(
        rfm_rows,
        eda,
        rfm_summary,
        rca,
        wald,
    )
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


def bar_row(
    label: str,
    value: float,
    count_text: str,
    tone: str = "gold",
) -> None:
    width = max(0, min(100, float(value)))
    cls = (
        "bar-negative"
        if tone == "negative"
        else "bar-positive"
        if tone == "positive"
        else "bar-gold"
    )
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
  --bg:#f7f5f0; --card:#ffffff; --ink:#20242d; --sub:#6f7480;
  --gold:#aa8351; --line:#ebe6dd; --green:#4e8d72; --red:#b65f63;
}
html, body, [class*="css"] {
  font-family:Pretendard,-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Noto Sans KR","Segoe UI",sans-serif;
}
[data-testid="stAppViewContainer"] {background:linear-gradient(180deg,#faf9f6 0%,var(--bg) 100%);color:var(--ink);}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:920px;padding-top:1.4rem;padding-bottom:5rem;}
.hero {position:relative;overflow:hidden;background:linear-gradient(135deg,#242a35,#303947);color:white;border-radius:28px;padding:1.7rem 1.8rem;margin-bottom:1rem;box-shadow:0 18px 45px rgba(25,29,36,.12);}
.hero:after {content:"⚖";position:absolute;right:1rem;top:-1.2rem;font-size:9rem;opacity:.08;}
.hero-eyebrow {font-size:.74rem;font-weight:800;letter-spacing:.16em;color:#dbc29b;}
.hero-title {font-size:2.35rem;font-weight:850;letter-spacing:-.05em;margin:.3rem 0 .45rem;}
.hero-sub {font-size:1rem;line-height:1.7;max-width:680px;color:#e4e7ec;}
.hero-chips {display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1rem;}
.hero-chip {font-size:.78rem;padding:.35rem .7rem;border:1px solid rgba(255,255,255,.18);border-radius:999px;background:rgba(255,255,255,.06);}
.section-title {font-size:1.28rem;font-weight:800;letter-spacing:-.03em;margin:.2rem 0 .25rem;}
.section-sub {font-size:.9rem;color:var(--sub);margin-bottom:.8rem;}
.result-hero {background:var(--card);border:1px solid var(--line);border-radius:22px;padding:1.2rem 1.25rem;box-shadow:0 10px 30px rgba(31,35,41,.06);}
.result-label {font-size:.74rem;font-weight:800;color:var(--gold);letter-spacing:.12em;text-transform:uppercase;}
.result-title {font-size:1.8rem;font-weight:850;letter-spacing:-.04em;margin:.25rem 0;}
.result-summary {font-size:1rem;line-height:1.65;color:#515762;}
.context-panel {background:linear-gradient(135deg,#fffdf8,#f7f2e9);border:1px solid #e7dbc9;border-radius:18px;padding:.9rem 1rem;margin:.75rem 0;}
.context-title {font-size:.82rem;font-weight:850;color:#856740;margin-bottom:.55rem;}
.context-chips {display:flex;gap:.48rem;flex-wrap:wrap;}
.context-chip {display:inline-flex;align-items:center;gap:.35rem;background:white;border:1px solid #e7dfd2;border-radius:999px;padding:.38rem .68rem;font-size:.79rem;}
.context-chip span {color:#7d818a;}.context-chip strong {color:#252a31;}
.context-empty {font-size:.86rem;color:#747982;line-height:1.5;}
.stat-card {background:var(--card);border:1px solid var(--line);border-radius:16px;padding:.85rem .9rem;min-height:100%;}
.stat-label {font-size:.76rem;color:var(--sub);font-weight:700;}
.stat-value {font-size:1.45rem;font-weight:850;margin-top:.15rem;}
.stat-note {font-size:.78rem;color:var(--sub);margin-top:.1rem;}
.bar-wrap {margin:.6rem 0 .8rem;}.bar-top {display:flex;justify-content:space-between;gap:1rem;font-size:.84rem;color:#555b65;margin-bottom:.28rem;}.bar-top strong {color:#2b3038;}
.bar-track {height:10px;border-radius:999px;background:#eeeae3;overflow:hidden;}.bar-fill {height:100%;border-radius:999px;}.bar-positive {background:#6f9c86;}.bar-negative {background:#ba7477;}.bar-gold {background:#b99a6d;}
.signal-grid {display:grid;grid-template-columns:repeat(3,1fr);gap:.55rem;margin-top:.6rem;}.signal-card {border:1px solid var(--line);border-radius:14px;padding:.7rem .75rem;background:#fcfbf8;}.signal-label {font-size:.75rem;color:var(--sub);}.signal-value {font-size:1.15rem;font-weight:850;margin-top:.05rem;}
.action-box {background:linear-gradient(135deg,#f4efe7,#faf8f4);border:1px solid #e4d6c2;border-radius:18px;padding:1rem 1.05rem;margin-top:.8rem;}.action-title {font-size:.85rem;color:#886a42;font-weight:800;margin-bottom:.45rem;}
[data-testid="stMetric"] {background:var(--card);border:1px solid var(--line);padding:.7rem .8rem;border-radius:15px;}[data-testid="stMetricValue"] {font-size:1.35rem;font-weight:850;}
.stButton>button,[data-testid="stFormSubmitButton"]>button {border-radius:999px;min-height:2.75rem;font-weight:700;}.stTextInput input {border-radius:13px!important;}[data-testid="stExpander"] {border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.7);}
@media(max-width:700px){.hero-title{font-size:1.9rem}.signal-grid{grid-template-columns:1fr}.block-container{padding-left:1rem;padding-right:1rem}}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <div class="hero-eyebrow">DIKE'S EYE · CONTEXT DECISION</div>
  <div class="hero-title">⚖️ 내 조건에 맞는 선택인지 따져봅니다</div>
  <div class="hero-sub">질문에서 조건을 읽고, 그 조건과 직접 연결되는 후기만 따로 비교합니다. 가격을 중요하게 봤다면 가격·가성비를, 조용함을 중요하게 봤다면 분위기·소음을 먼저 판단합니다.</div>
  <div class="hero-chips"><span class="hero-chip">Context · 내 조건</span><span class="hero-chip">Strength · 반복 강점</span><span class="hero-chip">Rashomon · 찬반</span><span class="hero-chip">Wald · 놓친 신호</span></div>
</div>
""",
    unsafe_allow_html=True,
)

DEFAULTS = {
    "intent": None,
    "parsed_context": {},
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

with st.container(border=True):
    st.markdown('<div class="section-title">무엇을 고민하고 있나요?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">대상과 조건을 한 문장으로 적어주세요. 가격, 조용함, 웨이팅, 배터리처럼 중요한 기준도 함께 적으면 그 항목을 직접 분석합니다.</div>', unsafe_allow_html=True)
    st.caption("예: 야키니쿠 하코 어때? 가격이 중요해")
    st.caption("예: 토요일 7시 소개팅인데 성수 어니언 어때? 조용한 곳이 중요해")
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "질문",
            placeholder="선택하려는 대상과 내 조건을 입력하세요",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "Dike에게 물어보기",
            type="primary",
            use_container_width=True,
        )

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
    st.markdown('<div class="section-title">질문에서 이렇게 읽었어요</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns([2.5, 1])
        with c1:
            st.markdown(f"**{kind_label} · {intent.get('target', '')}**")
            st.caption(f"원문 · {intent.get('original', '')}")
        with c2:
            st.metric(
                "질문 해석",
                f"{int(intent.get('parse_confidence', 0) * 100)}%",
            )
        render_context_chips(
            current_context(),
            kind,
            "질문에서 읽은 내 조건",
        )

    if kind == "restaurant" and not st.session_state.selected_target:
        if st.button(
            "NAVER에서 장소 확인",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("장소를 확인하고 있어요..."):
                    candidates, status = local_search(
                        intent.get("target") or intent.get("original", ""),
                        **naver_credentials(),
                    )
                st.session_state.search_status = status
                st.session_state.candidates = candidates or [
                    {
                        "title": intent.get("target") or intent.get("original", ""),
                        "category": "직접 입력",
                        "address": "",
                        "fallback": True,
                    }
                ]
            except Exception as exc:
                st.session_state.last_error = f"{type(exc).__name__}: {exc}"

        if st.session_state.candidates:
            labels = [
                f"{x.get('title', '')} · {x.get('category', '')} · {x.get('address', '')}"
                for x in st.session_state.candidates
            ]
            idx = st.radio(
                "이 장소가 맞나요?",
                range(len(labels)),
                format_func=lambda i: labels[i],
            )
            chosen = st.session_state.candidates[idx]
            if st.button(
                "이 장소가 맞아요",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.selected_target = {
                    "kind": "restaurant",
                    "name": chosen.get("title") or intent.get("target", ""),
                    "meta": chosen,
                }

    if kind == "product" and not st.session_state.selected_target:
        with st.form("product_confirm_form"):
            product_name = st.text_input(
                "제품명 / 모델명",
                key="target_edit",
            )
            product_confirmed = st.form_submit_button(
                "이 제품이 맞아요",
                type="primary",
                use_container_width=True,
            )
        if product_confirmed and product_name.strip():
            st.session_state.selected_target = {
                "kind": "product",
                "name": product_name.strip(),
                "meta": {},
            }

selected = st.session_state.selected_target
if selected and not st.session_state.analysis:
    st.divider()
    st.markdown('<div class="section-title">이 조건으로 분석할게요</div>', unsafe_allow_html=True)
    render_context_chips(
        current_context(),
        selected["kind"],
        "분석에 사용할 내 조건",
    )

    with st.expander("조건이 다르면 여기서 수정"):
        if selected["kind"] == "restaurant":
            a, b = st.columns(2)
            with a:
                st.text_input("방문 날짜/요일", key="ctx_day")
                st.text_input("시간", key="ctx_time")
            with b:
                st.text_input("목적", key="ctx_purpose")
                st.text_input("중요 조건", key="ctx_preference")
        else:
            a, b = st.columns(2)
            with a:
                st.text_input("사용 목적", key="ctx_purpose")
                st.text_input("중요 조건", key="ctx_preference")
            with b:
                st.text_input("사용 상황", key="ctx_day")
                st.text_input("추가 조건", key="ctx_time")

    if st.button(
        "이 조건으로 판단하기",
        type="primary",
        use_container_width=True,
    ):
        context = current_context()
        try:
            with st.spinner("내 조건에 직접 연결되는 후기와 장단점을 비교하고 있어요..."):
                analysis = run_decision(
                    selected["name"],
                    selected["kind"],
                    context,
                )
                report = build_user_report(
                    analysis,
                    selected["name"],
                    selected["kind"],
                )
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
    total_count = len(rows) + len(hidden_rows)
    result_context = a.get("context", {})

    st.divider()
    st.markdown(
        f"""
        <div class="result-hero">
          <div class="result-label">Dike's View</div>
          <div class="result-title">{r.get('headline', '')}</div>
          <div class="result-summary">{r.get('summary', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_context_chips(
        result_context,
        a.get("kind", "restaurant"),
        "이번 판단에 사용한 내 조건",
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f"<div class='stat-card'><div class='stat-label'>조건 적합도</div><div class='stat-value'>{d.get('fit_score', 0):.0f}/100</div><div class='stat-note'>장점과 내 조건 종합</div></div>",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"<div class='stat-card'><div class='stat-label'>판단 신뢰도</div><div class='stat-value'>{d.get('confidence', 0):.0f}%</div><div class='stat-note'>Evidence 품질 반영</div></div>",
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"<div class='stat-card'><div class='stat-label'>검토 Evidence</div><div class='stat-value'>{total_count}건</div><div class='stat-note'>Visible {len(rows)} + Hidden {len(hidden_rows)}</div></div>",
            unsafe_allow_html=True,
        )

    context_count = sum(1 for row in rows if row.get("context_aligned"))
    preference_count = sum(1 for row in rows if row.get("preference_aligned"))
    with st.container(border=True):
        st.markdown("**🎯 내 조건이 실제로 얼마나 반영됐나요?**")
        q1, q2, q3 = st.columns(3)
        q1.metric("전체 Visible", f"{len(rows)}건")
        q2.metric("내 조건 일치", f"{context_count}건")
        q3.metric("중요조건 직접 일치", f"{preference_count}건")
        if result_context.get("preference") and preference_count == 0:
            st.warning(
                f"'{result_context.get('preference')}' 조건과 직접 연결되는 Evidence를 충분히 찾지 못했어요. "
                "이 경우 다른 항목으로 대신 판단하지 않습니다."
            )

    st.markdown(
        '<div class="section-title" style="margin-top:1.3rem">먼저, 반복적으로 좋았던 점</div>',
        unsafe_allow_html=True,
    )
    positive_aspects = comps.get("positive_aspects", [])
    if positive_aspects:
        for item in positive_aspects[:3]:
            label = ASPECT_LABELS.get(
                str(item.get("aspect")),
                str(item.get("aspect")),
            )
            pos = int(item.get("positive_count", 0))
            neg = int(item.get("negative_count", 0))
            rate = float(item.get("positive_rate", 0.0)) * 100
            with st.container(border=True):
                st.markdown(f"**✨ {label}**")
                bar_row(
                    "긍정",
                    rate,
                    f"{pos}건 · {rate:.0f}%",
                    "positive",
                )
                st.caption(
                    f"긍정 {pos}건 / 부정 {neg}건으로 반복적으로 확인된 강점입니다."
                )
    else:
        st.info("반복적으로 확인되는 뚜렷한 긍정 강점은 아직 충분하지 않습니다.")

    conflicts = a.get("rca", {}).get("conflicts", [])
    if conflicts:
        p = conflicts[0]
        label = ASPECT_LABELS.get(
            str(p.get("aspect")),
            str(p.get("aspect")),
        )
        pos_count = int(p.get("positive_count", 0))
        neg_count = int(p.get("negative_count", 0))
        pos_rate = float(p.get("positive_rate", 0.0)) * 100
        neg_rate = float(p.get("negative_rate", 0.0)) * 100
        with st.container(border=True):
            st.markdown(f"**🎭 {label} — 의견이 갈린 지점**")
            bar_row("긍정", pos_rate, f"{pos_count}건 · {pos_rate:.0f}%", "positive")
            bar_row("부정", neg_rate, f"{neg_count}건 · {neg_rate:.0f}%", "negative")

    main_candidates = a.get("rca", {}).get("main_candidates", [])
    rca_point = main_candidates[0] if main_candidates else None
    if rca_point:
        label = ASPECT_LABELS.get(
            str(rca_point.get("aspect")),
            str(rca_point.get("aspect")),
        )
        ctx = str(rca_point.get("context") or "내 조건")
        base_total = int(rca_point.get("baseline_total_count", 0))
        base_neg = int(rca_point.get("baseline_negative_count", 0))
        ctx_total = int(rca_point.get("context_total_count", 0))
        ctx_neg = int(rca_point.get("context_negative_count", 0))
        base_rate = float(rca_point.get("baseline_negative_rate", 0.0)) * 100
        ctx_rate = float(rca_point.get("context_negative_rate", 0.0)) * 100
        diff = float(rca_point.get("lift", 0.0)) * 100

        with st.container(border=True):
            st.markdown(f"**🔎 {ctx} · {label} — 내 조건에서 달라진 부분**")
            c1, c2, c3 = st.columns(3)
            c1.metric("전체 부정", f"{base_rate:.0f}%", f"{base_neg}/{base_total}건")
            c2.metric("내 조건 부정", f"{ctx_rate:.0f}%", f"{ctx_neg}/{ctx_total}건")
            c3.metric("차이", f"{diff:+.1f}%p")
            bar_row("전체", base_rate, f"{base_rate:.0f}%", "gold")
            bar_row(
                "내 조건",
                ctx_rate,
                f"{ctx_rate:.0f}%",
                "negative" if diff > 0 else "positive",
            )
            if diff > 3:
                st.caption("내 조건에서는 부정 의견이 전체보다 더 자주 나타났습니다.")
            elif diff < -3:
                st.caption("내 조건에서는 부정 의견이 전체보다 적게 나타났습니다.")
            else:
                st.caption("내 조건과 전체 차이가 크지 않아 이 조건은 중립적으로 봅니다.")
    elif result_context.get("preference"):
        st.info(
            f"🔎 중요조건 '{result_context.get('preference')}'을 직접 비교할 Evidence가 아직 부족합니다. "
            "다른 조건의 결과로 대신 채우지 않았습니다."
        )

    signal_counts = a.get("wald", {}).get("signal_counts", {})
    if signal_counts:
        with st.container(border=True):
            st.markdown("**🕳️ 리뷰만 보면 놓칠 수 있는 신호**")
            top_signals = sorted(
                signal_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            cards = "".join(
                f"<div class='signal-card'><div class='signal-label'>{WALD_LABELS.get(k, k)}</div><div class='signal-value'>{v}건</div></div>"
                for k, v in top_signals
            )
            st.markdown(
                f"<div class='signal-grid'>{cards}</div>",
                unsafe_allow_html=True,
            )
            st.caption("실제 발생률이 아니라 검색된 이탈·실패 신호의 건수입니다.")

    if r.get("strengths") or r.get("findings"):
        with st.container(border=True):
            st.markdown("**📌 분석 결과를 한 번에 정리하면**")
            for item in (
                r.get("strengths", [])[:2]
                + r.get("findings", [])[:2]
            ):
                st.markdown(f"- {item}")

    st.markdown(
        '<div class="action-box"><div class="action-title">⚖️ Dike의 최종 제안</div>',
        unsafe_allow_html=True,
    )
    for item in r.get("recommendations", []):
        st.markdown(f"- **{item}**")
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("AI가 결과를 조금 더 자연스럽게 설명하기"):
        if st.button("설명 생성", use_container_width=True):
            with st.spinner("결과를 정리하고 있어요..."):
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

    with st.expander("분석 근거 자세히 보기"):
        st.markdown("##### 점수 구성")
        st.json({
            "weighted_sentiment": comps.get("weighted_sentiment"),
            "context_sentiment": comps.get("context_sentiment"),
            "positive_strength": comps.get("positive_strength"),
            "aligned_evidence_count": comps.get("aligned_evidence_count"),
            "preference_aligned_count": comps.get("preference_aligned_count"),
            "rca_risk": comps.get("rca_risk"),
            "wald_risk": comps.get("wald_risk"),
            "policy": d.get("policy", {}),
        })
        st.markdown("##### RCA · 사용자 조건")
        rca_df = pd.DataFrame(main_candidates)
        if not rca_df.empty:
            cols = [
                c
                for c in [
                    "aspect",
                    "context",
                    "analysis_scope",
                    "baseline_total_count",
                    "baseline_negative_count",
                    "context_total_count",
                    "context_negative_count",
                    "lift",
                    "confidence",
                ]
                if c in rca_df.columns
            ]
            st.dataframe(
                rca_df[cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("사용자 조건을 직접 비교할 만큼 충분한 Evidence가 없습니다.")

        st.markdown("##### 상위 Evidence")
        df = pd.DataFrame(rows)
        cols = [
            c
            for c in [
                "source",
                "retrieval_scope",
                "title",
                "aspects",
                "contexts",
                "context_aligned",
                "preference_aligned",
                "sentiment",
                "R",
                "F",
                "M",
                "priority",
            ]
            if c in df.columns
        ]
        if not df.empty:
            st.dataframe(
                df[cols].head(30),
                use_container_width=True,
                hide_index=True,
            )

    if st.button("새로운 질문하기", use_container_width=True):
        for key in list(DEFAULTS) + [
            "ctx_day",
            "ctx_time",
            "ctx_purpose",
            "ctx_preference",
            "target_edit",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

if st.session_state.last_error:
    st.error("처리 중 문제가 생겼어요. 입력 내용은 유지되어 있습니다.")
    with st.expander("오류 정보"):
        st.code(st.session_state.last_error)

st.divider()
st.caption(
    "Dike's Eye · 질문의 조건을 보존하고, 그 조건과 직접 연결되는 Evidence로 판단합니다."
)

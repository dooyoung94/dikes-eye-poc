"""Dike's Eye POC analysis modules and lightweight Streamlit UI patch."""

from __future__ import annotations

try:
    import streamlit as _st
except Exception:  # pragma: no cover
    _st = None


if _st is not None and not getattr(_st, "_dike_card_patch_installed", False):
    _st._dike_card_patch_installed = True

    _original_container = _st.container
    _original_set_page_config = _st.set_page_config
    _border_container_count = 0
    _style_injected = False

    _CARD_STYLE = r"""
    <style>
    /* Hero: keep the scale and gold frame, remove the oversized side laurels. */
    .hero-laurel {
        display: none !important;
    }

    /* Every bordered analysis block is a real parchment card, not text over Agora. */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, #fffaf0 0%, #f8ead2 100%) !important;
        border: 1px solid rgba(184, 139, 58, .72) !important;
        border-radius: 18px !important;
        padding: .9rem 1rem !important;
        box-shadow: 0 12px 30px rgba(59, 42, 22, .14) !important;
        backdrop-filter: none !important;
        color: #2b251d !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] p,
    div[data-testid="stVerticalBlockBorderWrapper"] strong,
    div[data-testid="stVerticalBlockBorderWrapper"] span,
    div[data-testid="stVerticalBlockBorderWrapper"] label,
    div[data-testid="stVerticalBlockBorderWrapper"] .bar > div {
        color: #2b251d !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }

    /* NAVER place confirmation: candidate names and radio labels get their own card. */
    div[data-testid="stRadio"] {
        background: linear-gradient(145deg, #fffaf0 0%, #f8ead2 100%) !important;
        border: 1px solid rgba(184, 139, 58, .72) !important;
        border-radius: 16px !important;
        padding: .82rem 1rem !important;
        margin: .45rem 0 .75rem !important;
        box-shadow: 0 10px 24px rgba(59, 42, 22, .13) !important;
        color: #2b251d !important;
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] p,
    div[data-testid="stRadio"] span,
    div[data-testid="stRadio"] [role="radiogroup"] {
        color: #2b251d !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }

    [data-testid="stExpander"] {
        background: #fffdf7 !important;
        border-color: rgba(184, 139, 58, .48) !important;
    }

    /* Top conversation cards. */
    .st-key-question_panel,
    .st-key-intent_panel {
        position: relative;
        background: linear-gradient(145deg, #fffaf0 0%, #f8ead2 100%) !important;
        border: 1px solid rgba(184, 139, 58, .78) !important;
        border-radius: 22px !important;
        padding: 1.15rem 1.2rem 1.05rem !important;
        box-shadow:
            0 16px 38px rgba(59, 42, 22, .18),
            inset 0 0 0 1px rgba(255, 255, 255, .58) !important;
        overflow: hidden;
        isolation: isolate;
    }

    .st-key-question_panel::before,
    .st-key-intent_panel::before {
        content: "";
        position: absolute;
        inset: 0;
        z-index: -1;
        pointer-events: none;
        opacity: .16;
        background:
            radial-gradient(circle at 18% 8%, rgba(212,175,55,.18), transparent 34%),
            linear-gradient(120deg, transparent 0 47%, rgba(155,121,62,.08) 50%, transparent 53%);
    }

    /* Avoid double cards inside the two keyed conversation cards. */
    .st-key-question_panel [data-testid="stVerticalBlockBorderWrapper"],
    .st-key-intent_panel [data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    .st-key-question_panel [data-testid="stForm"] {
        margin-top: .45rem !important;
        padding: .72rem !important;
        border: 1px solid rgba(190, 147, 70, .48) !important;
        border-radius: 15px !important;
        background: rgba(255, 253, 248, .86) !important;
        box-shadow: none !important;
    }

    .st-key-question_panel .section-title,
    .st-key-intent_panel .section-title {
        display: block !important;
        width: fit-content;
        margin: 0 0 .45rem !important;
        padding: 0 !important;
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        color: #241d15 !important;
    }

    .st-key-question_panel .section-sub,
    .st-key-intent_panel .section-sub,
    .st-key-question_panel [data-testid="stCaptionContainer"],
    .st-key-intent_panel [data-testid="stCaptionContainer"] {
        color: #5a5145 !important;
        opacity: 1 !important;
        text-shadow: none !important;
    }

    .st-key-question_panel [data-testid="stCaptionContainer"] {
        background: rgba(255, 250, 239, .64) !important;
        border-left: 2px solid rgba(184, 139, 58, .42) !important;
        border-radius: 0 8px 8px 0 !important;
        padding: .25rem .48rem !important;
        margin: .18rem 0 !important;
    }

    .st-key-intent_panel .context {
        margin: .65rem 0 0 !important;
        background: #fffdf7 !important;
        border: 1px solid rgba(190,147,70,.50) !important;
        box-shadow: 0 5px 14px rgba(72, 52, 29, .06) !important;
    }

    .st-key-intent_panel p,
    .st-key-intent_panel strong,
    .st-key-question_panel p,
    .st-key-question_panel strong {
        color: #2b251d !important;
    }

    @media (max-width: 700px) {
        .st-key-question_panel,
        .st-key-intent_panel {
            padding: .82rem .78rem .78rem !important;
            border-radius: 17px !important;
            box-shadow: 0 10px 24px rgba(59,42,22,.16) !important;
        }

        .st-key-question_panel [data-testid="stForm"] {
            padding: .55rem !important;
        }

        div[data-testid="stRadio"],
        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: .7rem .72rem !important;
            border-radius: 14px !important;
        }
    }
    </style>
    """

    def _patched_set_page_config(*args, **kwargs):
        global _border_container_count, _style_injected
        _border_container_count = 0
        _style_injected = False
        return _original_set_page_config(*args, **kwargs)

    def _patched_container(*args, **kwargs):
        global _border_container_count, _style_injected

        is_bordered = bool(kwargs.get("border", False))
        if is_bordered and not kwargs.get("key"):
            _border_container_count += 1
            if _border_container_count == 1:
                kwargs["key"] = "question_panel"
            elif _border_container_count == 2:
                kwargs["key"] = "intent_panel"

        if is_bordered and not _style_injected:
            _style_injected = True
            _st.markdown(_CARD_STYLE, unsafe_allow_html=True)

        return _original_container(*args, **kwargs)

    _st.set_page_config = _patched_set_page_config
    _st.container = _patched_container

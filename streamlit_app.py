import streamlit as st

st.set_page_config(page_title="Dike's Eye POC", page_icon="⚖️", layout="centered")

st.markdown(
    """
    <style>
    .block-container {max-width: 860px; padding-top: 2rem; padding-bottom: 4rem;}
    .hero {border:1px solid rgba(128,128,128,.2); border-radius:22px; padding:1.2rem 1.4rem; margin-bottom:1rem;}
    .soft {opacity:.7;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="soft">Bias-aware Review Decision Agent · POC</div>
      <h1>⚖️ Dike's Eye</h1>
      <div>지금 단계에서는 Streamlit 입력과 상태 유지가 정상인지 먼저 확인합니다.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        ("assistant", "어디를 갈지, 무엇을 살지 고민 중인가요? 질문을 입력해 주세요.")
    ]

for role, text in st.session_state.messages:
    if role == "assistant":
        st.markdown(f"**⚖️ Dike's Eye**  \n{text}")
    else:
        st.markdown(f"**🙂 나**  \n{text}")

with st.form("question_form", clear_on_submit=True):
    question = st.text_input(
        "질문",
        placeholder="예: 토요일 7시 성수 어니언, 소개팅으로 괜찮아?",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("질문 보내기", type="primary", use_container_width=True)

if submitted:
    text = question.strip()
    if text:
        st.session_state.messages.append(("user", text))
        st.session_state.messages.append(
            ("assistant", f"입력을 정상적으로 받았어요: **{text}**\n\n다음 단계에서 NAVER 후보 검색과 분석 엔진을 연결할 수 있습니다.")
        )
        st.success("질문 입력과 session_state 저장이 정상 동작했습니다.")
    else:
        st.warning("질문을 입력해 주세요.")

st.divider()
st.caption("POC Step 1 · 외부 API / OpenAI / EDA / RFM / RCA 미사용")

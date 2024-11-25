import streamlit as st

intro_page = st.Page(
    "application_pages/intro_page.py", title="Introduction", icon=":material/guardian:"
)
chat_page = st.Page(
    "application_pages/chat_app.py", title="Chat", icon=":material/robot:"
)
evaluation_page = st.Page(
    "application_pages/evaluation_app.py",
    title="Evaluation",
    icon=":material/monitoring:",
)
page_navigation = st.navigation([intro_page, chat_page, evaluation_page])
st.set_page_config(page_title="Guardrails Genie", page_icon=":material/guardian:")
page_navigation.run()
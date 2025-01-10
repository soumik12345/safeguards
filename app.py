import streamlit as st

intro_page = st.Page(
    "application_pages/intro_page.py", title="Authenticate", icon=":material/guardian:"
)
chat_page = st.Page(
    "application_pages/chat_app.py",
    title="Playground",
    icon=":material/sports_esports:",
)
page_navigation = st.navigation([intro_page, chat_page])
st.set_page_config(page_title="Guardrails Genie", page_icon=":material/guardian:")
page_navigation.run()

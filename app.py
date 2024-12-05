import streamlit as st

intro_page = st.Page(
    "application_pages/intro_page.py", title="Introduction", icon=":material/guardian:"
)
chat_page = st.Page(
    "application_pages/chat_app.py",
    title="Playground",
    icon=":material/sports_esports:",
)
evaluation_page = st.Page(
    "application_pages/evaluation_app.py",
    title="Evaluation",
    icon=":material/monitoring:",
)
# train_classifier_page = st.Page(
#     "application_pages/train_classifier.py",
#     title="Train Classifier",
#     icon=":material/fitness_center:",
# )
# llama_guard_fine_tuning_page = st.Page(
#     "application_pages/llama_guard_fine_tuning.py",
#     title="Fine-Tune LLama Guard",
#     icon=":material/star:",
# )
page_navigation = st.navigation(
    [
        intro_page,
        chat_page,
        evaluation_page,
        # train_classifier_page,
        # llama_guard_fine_tuning_page,
    ]
)
st.set_page_config(page_title="Guardrails Genie", page_icon=":material/guardian:")
page_navigation.run()

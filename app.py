import streamlit as st
import weave
from dotenv import load_dotenv

from guardrails_genie.llm import OpenAIModel

load_dotenv()
weave.init(project_name="guardrails-genie")


openai_model = st.sidebar.selectbox("OpenAI LLM", ["", "gpt-4o-mini", "gpt-4o"])
chat_condition = openai_model != ""


if chat_condition:
    st.title("Guardrails Genie")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    llm_model = OpenAIModel(model_name="gpt-4o-mini")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response, call = llm_model.predict.call(
            llm_model, user_prompts=prompt, messages=st.session_state.messages
        )
        response = response.choices[0].message.content
        response += f"\n\n---\n[Explore in Weave]({call.ui_url})"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

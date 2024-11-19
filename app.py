import streamlit as st
import weave
from dotenv import load_dotenv

from guardrails_genie.llm import OpenAIModel

load_dotenv()
weave.init(project_name="guardrails-genie")

openai_model = st.sidebar.selectbox("OpenAI LLM", ["", "gpt-4o-mini", "gpt-4o"])
chat_condition = openai_model != ""

# Use session state to track if the chat has started
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

# Start chat when button is pressed
if st.sidebar.button("Start Chat") and chat_condition:
    st.session_state.chat_started = True

# Display chat UI if chat has started
if st.session_state.chat_started:
    st.title("Guardrails Genie")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    llm_model = OpenAIModel(model_name=openai_model)

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

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response + f"\n\n---\n[Explore in Weave]({call.ui_url})")
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

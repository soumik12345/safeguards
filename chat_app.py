import importlib

import streamlit as st
import weave
from dotenv import load_dotenv

from guardrails_genie.guardrails import GuardrailManager
from guardrails_genie.llm import OpenAIModel

load_dotenv()
weave.init(project_name="guardrails-genie")

if "guardrails" not in st.session_state:
    st.session_state.guardrails = []
if "guardrail_names" not in st.session_state:
    st.session_state.guardrail_names = []
if "guardrails_manager" not in st.session_state:
    st.session_state.guardrails_manager = None


def initialize_guardrails():
    st.session_state.guardrails = []
    for guardrail_name in st.session_state.guardrail_names:
        if guardrail_name == "PromptInjectionSurveyGuardrail":
            survey_guardrail_model = st.sidebar.selectbox(
                "Survey Guardrail LLM", ["", "gpt-4o-mini", "gpt-4o"]
            )
            if survey_guardrail_model:
                st.session_state.guardrails.append(
                    getattr(
                        importlib.import_module("guardrails_genie.guardrails"),
                        guardrail_name,
                    )(llm_model=OpenAIModel(model_name=survey_guardrail_model))
                )
        else:
            st.session_state.guardrails.append(
                getattr(
                    importlib.import_module("guardrails_genie.guardrails"),
                    guardrail_name,
                )()
            )
    st.session_state.guardrails_manager = GuardrailManager(
        guardrails=st.session_state.guardrails
    )


openai_model = st.sidebar.selectbox(
    "OpenAI LLM for Chat", ["", "gpt-4o-mini", "gpt-4o"]
)
chat_condition = openai_model != ""

guardrails = []

guardrail_names = st.sidebar.multiselect(
    label="Select Guardrails",
    options=[
        cls_name
        for cls_name, cls_obj in vars(
            importlib.import_module("guardrails_genie.guardrails")
        ).items()
        if isinstance(cls_obj, type) and cls_name != "GuardrailManager"
    ],
)
st.session_state.guardrail_names = guardrail_names

# Use session state to track if the chat has started
if "chat_started" not in st.session_state:
    st.session_state.chat_started = False

# Start chat when button is pressed
if st.sidebar.button("Start Chat") and chat_condition:
    st.session_state.chat_started = True

# Display chat UI if chat has started
if st.session_state.chat_started:
    with st.sidebar.status("Initializing Guardrails..."):
        initialize_guardrails()

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

        guardrails_response, call = st.session_state.guardrails_manager.guard.call(
            st.session_state.guardrails_manager, prompt=prompt
        )

        if guardrails_response["safe"]:
            response, call = llm_model.predict.call(
                llm_model, user_prompts=prompt, messages=st.session_state.messages
            )
            response = response.choices[0].message.content

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response + f"\n\n---\n[Explore in Weave]({call.ui_url})")
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.error("Guardrails detected an issue with the prompt.")
            for alert in guardrails_response["alerts"]:
                st.error(f"{alert['guardrail_name']}: {alert['response']}")
            st.error(f"For details, explore in Weave at {call.ui_url}")

import os

import rich
import wandb
from fasthtml.core import serve
from monsterui.core import FastHTML, P, Theme

from safeguards.app.components import (
    GuardrailsPlaygroundPage,
    SafeGuardsNavBar,
    SettingsPage,
)
from safeguards.app.components.commons import AlertStatusNotification
from safeguards.app.state import AppState, SettingState
from safeguards.llm import OpenAIModel

app = FastHTML(hdrs=Theme.blue.headers())
route = app.route
app_state = AppState()


@app.get("/")
def landing_page():
    return SafeGuardsNavBar()


@app.get("/settings")
def settings_page():
    return SettingsPage()


@app.post("/save_settings")
def save_settings(
    wandb_username: str, wandb_project: str, wandb_api_key: str, openai_api_key: str
):
    success = True
    status_messages = []
    if wandb_username == "":
        success = False
        status_messages.append("W&B username cannot be blank.")
    if wandb_project == "":
        success = False
        status_messages.append("W&B project cannot be blank.")
    if wandb_api_key == "":
        success = False
        status_messages.append("W&B API key cannot be blank.")
    if wandb_api_key == "":
        success = False
        status_messages.append("W&B API key cannot be blank.")
    try:
        wandb_initialization_status = wandb.login(key=wandb_api_key, verify=True)
        if not wandb_initialization_status:
            success = False
            status_messages.append("W&B initialization failed, invalid credentials.")
    except Exception as e:
        success = False
        status_messages.append(str(e))
    os.environ["OPENAI_API_KEY"] = openai_api_key
    if success:
        status_messages.append("Settings saved successfully!")
        app_state.settings_state = SettingState(
            wandb_username=wandb_username,
            wandb_project=wandb_project,
            wandb_api_key=wandb_api_key,
            openai_api_key=openai_api_key,
        )
    rich.print(status_messages)
    return AlertStatusNotification(message="\n".join(status_messages), success=success)


@app.get("/guardrails_playground")
def guardrails_playground_page():
    return GuardrailsPlaygroundPage(state=app_state)


@app.post("/playground_llm_selection_update")
async def guardrails_playground_llm_selection(playground_llm_selection: str):
    app_state.llm_model = OpenAIModel(model_name=playground_llm_selection)
    rich.print("OpenAI Model initialized!!!")
    return P("")


serve()

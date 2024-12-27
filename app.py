import os

import wandb
from fasthtml.common import FastHTML, FileResponse, Link, serve

from safeguards.app.components.commons import SafeGuardsNavBar
from safeguards.app.components.landing_page import SafeGuardsLanding
from safeguards.app.components.settings import SettingsForm, SettingsModal
from safeguards.app.state import AppState, SettingState
from safeguards.app.tailwind import add_daisy_and_tailwind

app = FastHTML(hdrs=Link(rel="stylesheet", href="app.css", type="text/css"))
app.static_route_exts(static_path="public")
add_daisy_and_tailwind(app)
route = app.route
app_state = AppState()


@app.get("/{fname:path}.{ext:static}")
def get_static(fname: str, ext: str):
    return FileResponse(f"public/{fname}.{ext}")


@app.get("/")
def get_index():
    return (SafeGuardsNavBar(), SafeGuardsLanding())


@app.get("/settings")
def get_settings():
    return (SafeGuardsNavBar(), SettingsForm())


@app.post("/save_settings")
def save_settings(
    wandb_entity: str, wandb_project: str, wandb_api_key: str, openai_api_key: str
):
    success = True
    status_messages = []
    if wandb_entity == "":
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
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_api_key=wandb_api_key,
            openai_api_key=openai_api_key,
        )
    return SettingsModal(message="\n".join(status_messages), success=success)


serve()

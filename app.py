import os

import rich
import rich.pretty
import wandb
from fasthtml.common import FastHTML, FileResponse, serve

from safeguards.app.components.commons import SafeGuardsNavBar, StatusModal
from safeguards.app.components.guardrails_playground import GuardrailsPlayGroundPage
from safeguards.app.components.landing_page import SafeGuardsLanding
from safeguards.app.components.settings import SettingsForm
from safeguards.app.state import AppState, SettingState
from safeguards.app.tailwind import add_daisy_and_tailwind
from safeguards.llm import OpenAIModel

app = FastHTML()
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
        app_state.is_settings_saved = True
    return StatusModal(message="\n".join(status_messages), success=success)


@app.get("/guardrails_playground")
def get_guardrails_playground():
    return (SafeGuardsNavBar(), GuardrailsPlayGroundPage(state=app_state))


@app.post("/playground_llm_selection_update")
async def post_playground_llm_selection_update(playground_llm_selection: str):
    try:
        app_state.llm_model = OpenAIModel(model_name=playground_llm_selection)
        return StatusModal(
            f"Initialized {playground_llm_selection} as Playground LLM", success=True
        )
    except Exception as e:
        return StatusModal(
            f"Failed to initialize {playground_llm_selection} as Playground LLM! "
            + str(e),
            success=False,
        )


@app.post("/playground_guardrail_selection_regexentityrecognition")
async def post_playground_guardrail_selection_regexentityrecognition(
    regexentityrecognition: str,
):
    rich.print(regexentityrecognition)


@app.post("/playground_guardrail_selection_transformersentityrecognition")
async def post_playground_guardrail_selection_transformersentityrecognition(
    transformersentityrecognition: str,
):
    rich.print(transformersentityrecognition)


@app.post("/playground_guardrail_selection_promptinjectionclassifier")
async def post_playground_guardrail_selection_promptinjectionclassifier(
    promptinjectionclassifier: str,
):
    rich.print(promptinjectionclassifier)


@app.post("/playground_guardrail_selection_promptinjectionllm")
async def post_playground_guardrail_selection_promptinjectionllm(
    promptinjectionllm: str,
):
    rich.print(promptinjectionllm)


@app.post("/playground_guardrail_selection_openaiprivilegeescalation")
async def post_playground_guardrail_selection_openaiprivilegeescalation(
    openaiprivilegeescalation: str,
):
    rich.print(openaiprivilegeescalation)


@app.post("/playground_guardrail_selection_sqlinjection")
async def post_playground_guardrail_selection_sqlinjection(sqlinjection: str):
    rich.print(sqlinjection)


@app.post("/playground_guardrail_selection_secretsdetection")
async def post_playground_guardrail_selection_secretsdetection(secretsdetection: str):
    rich.print(secretsdetection)


serve()

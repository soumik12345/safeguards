import rich
from fasthtml.core import serve
from monsterui.core import FastHTML, Theme

from safeguards.app.components import SafeGuardsNavBar, SettingsPage

app = FastHTML(hdrs=Theme.blue.headers())
route = app.route


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
    rich.print(f"{wandb_username=}")
    rich.print(f"{wandb_project=}")
    rich.print(f"{wandb_api_key=}")
    rich.print(f"{openai_api_key=}")


serve()

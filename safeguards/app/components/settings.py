from monsterui.core import A, Div, Form, P
from monsterui.daisy import Loading
from monsterui.franken import DividerSplit, LabelInput, TextFont, UkFormSection

from .navbar import SafeGuardsNavBar


def SettingsHeading():
    return Div(cls="space-y-5")(
        P("Manage your W&B Project Settings and API keys.", cls=TextFont.muted_lg),
        DividerSplit(),
    )


def SettingsForm():
    return (
        Form(
            UkFormSection(
                "W&B Account Settings and API Keys",
                Div(cls="space-y-2")(
                    LabelInput("W&B Entity", placeholder="", id="wandb_username"),
                    P("Weights & Biases entity name", cls=TextFont.muted_sm),
                ),
                Div(cls="space-y-2")(
                    LabelInput("W&B Project", placeholder="", id="wandb_project"),
                    P("Weights & Biases project name", cls=TextFont.muted_sm),
                ),
                Div(cls="space-y-2")(
                    LabelInput(
                        "W&B API Key",
                        placeholder="",
                        id="wandb_api_key",
                        type="password",
                    ),
                    P(
                        "You can get your Weights & Biases API Key from ",
                        A("wandb.ai/authorize", href="https://wandb.ai/authorize"),
                        cls=TextFont.muted_sm,
                    ),
                ),
                Div(cls="space-y-2")(
                    LabelInput(
                        "OpenAI API Key",
                        placeholder="",
                        id="openai_api_key",
                        type="password",
                    ),
                    P(
                        "You can get your OpenAI API Key from ",
                        A(
                            "platform.openai.com/api-keys",
                            href="https://platform.openai.com/api-keys",
                        ),
                        cls=TextFont.muted_sm,
                    ),
                ),
                button_txt="Save Settings",
            ),
            hx_post="/save_settings",
            hx_target="#settings_status",
            hx_indicator="#loading",
        ),
        Div(id="settings_status"),
        Loading(id="loading", htmx_indicator=True),
    )


def SettingsPage():
    return Div(cls="p-6 lg:p-10")(
        SafeGuardsNavBar(),
        SettingsForm(),
    )

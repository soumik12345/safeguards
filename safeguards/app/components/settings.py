from fasthtml.components import H2, Button, Div, Form, Input, Label, Span


def SettingsFormInput(label: str, id: str, input_type: str = "text"):
    return (
        Div(
            Label(Span(label, cls="label-text"), cls="label"),
            Label(
                Input(type=input_type, id=id, cls="grow"),
                cls="input input-bordered flex items-center gap-2",
            ),
            cls="form-control",
        ),
    )


def SettingsForm():
    return Div(
        Div(
            Div(
                H2("Settings", cls="card-title text-2xl font-bold mb-6"),
                Form(
                    SettingsFormInput("W&B Entity", "wandb_entity"),
                    SettingsFormInput("W&B Project", "wandb_project"),
                    SettingsFormInput(
                        "W&B API Key", "wandb_api_key", input_type="password"
                    ),
                    SettingsFormInput(
                        "OpenAI API Key", "openai_api_key", input_type="password"
                    ),
                    Div(Button("Save", cls="btn btn-primary"), cls="form-control mt-6"),
                    hx_post="/save_settings",
                    hx_target="#settings_status",
                ),
                Div(id="settings_status"),
                cls="card-body",
            ),
            cls="card w-96 bg-base-100 shadow-xl",
        ),
        cls="bg-base-200 flex items-center justify-center min-h-screen",
    )

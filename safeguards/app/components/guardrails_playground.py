import importlib

from monsterui.franken import (
    H2,
    H3,
    Div,
    DivCentered,
    DividerLine,
    Form,
    FormLabel,
    Grid,
    LabelCheckboxX,
    LabelUkSelect,
    Option,
)

from ..state import AppState
from .commons import AlertStatusNotification
from .navbar import SafeGuardsNavBar


def PlaygroundHeaders():
    return (
        Grid(
            DivCentered(
                DividerLine(y_space=2),
                H3("Playgorund Settings"),
                DividerLine(y_space=2),
            ),
            DivCentered(DividerLine(y_space=2), H3("Chat"), DividerLine(y_space=2)),
            cols=2,
        ),
    )


def PlaygroundLLMModelSelection():
    return (
        Form(
            LabelUkSelect(
                map(Option, ["gpt-4o", "gpt-4o-mini"]),
                label="Select Chat LLM",
                name="playground_llm_selection",
                hx_post="/playground_llm_selection_update",
                hx_target="#playground_llm_selection_target",
            ),
        ),
        Div(id="playground_llm_selection_target"),
    )


def PlaygroundGuardrailsCheckboxGroup():
    guardrail_checkboxes = [
        Div(LabelCheckboxX(label=cls_name, id=cls_name))
        for cls_name, cls_obj in vars(
            importlib.import_module("safeguards.guardrails")
        ).items()
        if not cls_name.startswith("__") and cls_name.endswith("Guardrail")
    ]
    return Form(FormLabel("Select Guardrails"), *guardrail_checkboxes)


def PlaygroundSettingsInterface():
    return (PlaygroundLLMModelSelection(), PlaygroundGuardrailsCheckboxGroup())


def PlayGroundChatInterface():
    return Div(H3("Chat Interface"))


def GuardrailsPlaygroundPage(state: AppState):
    content = [SafeGuardsNavBar()]
    if state.settings_state is None:
        content.append(
            AlertStatusNotification(
                "Please save your W&B account settings and API Keys.", success=False
            )
        )
    else:
        content.extend(
            [
                Div(cls="space-y-6 py-6")(Div(H2("Guardrails Playground"))),
                # PlaygroundHeaders(),
                Grid(
                    Div(cls="space-y-6 py-6")(PlaygroundSettingsInterface()),
                    Div(cls="space-y-6 py-6")(PlayGroundChatInterface()),
                    cols=2,
                ),
            ]
        )
    return Div(cls="p-6 lg:p-10")(*content)

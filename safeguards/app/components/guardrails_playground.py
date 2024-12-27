import importlib

from fasthtml.common import Div, Form, Input, Option, Select, Span

from ..state import AppState
from .commons import StatusModal


def GuardrailsPlaygroundLLMSelection():
    return Div(
        Form(
            Div(
                Span("Select the LLM for Playground Chat", cls="label-text"),
                cls="label",
            ),
            Select(
                Option(""),
                Option("gpt-4o"),
                Option("gpt-4o-mini"),
                cls="select select-bordered",
                name="playground_llm_selection",
                hx_post="/playground_llm_selection_update",
                hx_target="#playground_llm_selection_target",
            ),
            Div(id="playground_llm_selection_target"),
            cls="form-control w-full max-w-xs",
        ),
        cls="container mx-auto p-6",
    )


def GuardrailsPlaygroundCheckbox(label: str):
    name = label.lower().removesuffix("guardrail")
    return Form(
        Input(
            type="checkbox",
            cls="checkbox checkbox-primary",
            name=name,
            hx_post=f"/playground_guardrail_selection_{name}",
        ),
        Span(label),
        cls="flex items-center gap-2",
    )


def GuardrailsPlaygroundGuardrailSelection():
    guardrail_checkboxes = [
        GuardrailsPlaygroundCheckbox(label=cls_name)
        for cls_name, cls_obj in vars(
            importlib.import_module("safeguards.guardrails")
        ).items()
        if not cls_name.startswith("__") and cls_name.endswith("Guardrail")
    ]
    return Div(
        Div(
            Div(
                Span("Select Guardrails", cls="label-text"),
                cls="label",
            ),
            *guardrail_checkboxes,
            cls="flex flex-col gap-4 p-6 rounded-lg",
        ),
        cls="container mx-auto p-6",
    )


def GuardrailsPlaygroundSettings():
    return Div(
        GuardrailsPlaygroundLLMSelection(),
        GuardrailsPlaygroundGuardrailSelection(),
        cls="flex flex-col",
    )


def GuardrailsPlayGroundPage(state: AppState):
    if not state.is_settings_saved:
        return StatusModal(
            message="Please save your W&B account settings and API Keys.", success=False
        )
    return Div(
        GuardrailsPlaygroundSettings(), Div("Col 2"), cls="grid grid-cols-2 gap-4 p-4"
    )

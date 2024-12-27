from fasthtml.common import Div, Form, Option, Select, Span

from ..state import AppState
from .commons import StatusModal


def GuardrailsPlaygroundLLMSelection():
    return Form(
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
    )


def GuardrailsPlaygroundSettings():
    return [GuardrailsPlaygroundLLMSelection()]


def GuardrailsPlayGroundPage(state: AppState):
    if not state.is_settings_saved:
        return StatusModal(
            message="Please save your W&B account settings and API Keys.", success=False
        )
    return Div(
        *GuardrailsPlaygroundSettings(), Div("Col 2"), cls="grid grid-cols-2 gap-4 p-4"
    )

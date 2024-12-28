from fasthtml.common import Div

from ...state import AppState
from .settings import GuardrailsPlaygroundSettings


def GuardrailsPlayGroundPage(state: AppState):
    # if not state.is_settings_saved:
    #     return StatusModal(
    #         message="Please save your W&B account settings and API Keys.", success=False
    #     )
    return Div(
        GuardrailsPlaygroundSettings(), Div("Col 2"), cls="grid grid-cols-2 gap-4 p-4"
    )

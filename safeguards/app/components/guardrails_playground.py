from monsterui.franken import H3, Div

from ..state import AppState
from .commons import AlertStatusNotification
from .navbar import SafeGuardsNavBar


def GuardrailsPlaygroundPage(state: AppState):
    content = [SafeGuardsNavBar()]
    if state.settings_state is None:
        content.append(
            AlertStatusNotification(
                "Please save your W&B account settings and API Keys.", success=False
            )
        )
    else:
        content.append(Div(cls="space-y-6 py-6")(Div(H3("Guardrails Playground"))))
    return Div(cls="p-6 lg:p-10")(*content)

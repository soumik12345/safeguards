from monsterui.franken import H2, H3, Div, DivCentered, DividerLine, Grid

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
        content.extend(
            [
                Div(cls="space-y-6 py-6")(Div(H2("Guardrails Playground"))),
                Grid(
                    DivCentered(
                        DividerLine(y_space=2),
                        H3("Playgorund Settings"),
                        DividerLine(y_space=2),
                    ),
                    DivCentered(
                        DividerLine(y_space=2), H3("Chat"), DividerLine(y_space=2)
                    ),
                    cols=2,
                ),
            ]
        )
    return Div(cls="p-6 lg:p-10")(*content)

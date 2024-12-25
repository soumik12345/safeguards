from monsterui.core import A, Li
from monsterui.franken import (
    Div,
    NavBarCenter,
    NavBarContainer,
    NavBarLSide,
    NavBarNav,
    NavBarNavContainer,
    NavBarParentIcon,
    NavBarRSide,
)


def SafeGuardsNavBar():
    return NavBarContainer(
        NavBarLSide(
            A(Div("SafeGuards", cls="navbar-title"), href="/"),
        ),
        NavBarCenter(
            NavBarNav(
                Li(
                    A("Guardrails", href=""),
                    NavBarNavContainer(
                        Li(A("Playground", href="/guardrails_playground")),
                        Li(A("Evaluations", href="/guardrails_evaluations")),
                    ),
                ),
            )
        ),
        NavBarRSide(
            NavBarNav(
                Li(
                    A("Profile", NavBarParentIcon(), href=""),
                    NavBarNavContainer(Li(A("Settings", href="/settings"))),
                ),
            )
        ),
    )

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
                    A("Parent", href=""),
                    NavBarNavContainer(
                        Li(A("Item", href="")),
                        Li(A("Item", href="")),
                    ),
                ),
                Li(A("Item", href="")),
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

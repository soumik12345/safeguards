from monsterui.core import A, Li
from monsterui.franken import (
    NavBarContainer,
    NavBarLSide,
    NavBarCenter,
    NavBarRSide,
    NavBarNav,
    NavBarNavContainer,
    Div,
    NavBarParentIcon,
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
                    A("DropDown", NavBarParentIcon(), href=""),
                    NavBarNavContainer(Li(A("Item", href="")), Li(A("Item", href=""))),
                ),
            )
        ),
    )

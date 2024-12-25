from monsterui.core import A, Li
from monsterui.franken import Container, TabContainer


def SafeGuardsNavBar():
    return Container(
        TabContainer(
            Li(A("Active", href="#", cls="uk-active")),
            Li(A("Item", href="#")),
            Li(A("Item", href="#")),
            Li(A("Disabled", href="#", cls="uk-disabled")),
            uk_switcher="connect: #component-nav; animation: uk-animation-fade",
            alt=True,
        ),
    )

import reflex as rx

from .components import navbar, playground_ui, settings_form
from .state import SettingsState


def settings() -> rx.Component:
    return rx.box(
        navbar(),
        rx.vstack(
            settings_form(),
            align="center",
            justify="center",
            padding="2%",
        ),
    )


def index() -> rx.Component:
    return rx.cond(
        SettingsState.authentication_successful,
        playground(),
        rx.box(
            navbar(),
            rx.vstack(
                rx.heading("Welcome to Safeguards!", size="9"),
                rx.text(
                    "A platform to implement crucial safety checks in LLM applications",
                    size="5",
                ),
                align="center",
                spacing="5",
                justify="center",
                min_height="85vh",
            ),
        ),
    )


def playground() -> rx.Component:
    return rx.cond(
        SettingsState.authentication_successful,
        rx.box(
            navbar(),
            playground_ui(),
        ),
        settings(),
    )

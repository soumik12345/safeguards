from typing import Optional

import reflex as rx

from ..state import SettingsState


def settings_header() -> rx.Component:
    return (
        rx.center(
            rx.heading(
                "Settings",
                size="6",
                as_="h2",
                text_align="center",
                width="100%",
            ),
            direction="column",
            spacing="5",
            width="100%",
        ),
    )


def input_field(
    label: str,
    placeholder: Optional[str] = None,
    type: str = "text",
    on_blur: Optional[rx.EventHandler] = None,
) -> rx.Component:
    return rx.vstack(
        rx.text(
            label,
            size="3",
            weight="medium",
            text_align="left",
            width="100%",
        ),
        rx.input(
            placeholder=placeholder,
            type=type,
            size="3",
            width="100%",
            on_blur=on_blur,
        ),
        justify="start",
        spacing="2",
        width="100%",
    )


def error_message(text: str, color: str = "red") -> rx.Component:
    return rx.box(
        rx.text(
            text,
            color=color,
            size="4",
            weight="bold",
            text_align="center",
        ),
        border="1px solid",
        border_color=color,
        padding="1em",
        border_radius="md",
        width="100%",
        max_width="28em",
        margin="auto",
    )


def settings_form() -> rx.Component:
    return rx.card(
        rx.vstack(
            settings_header(),
            input_field(
                "W&B Project Name",
                "aiwatch",
                on_blur=SettingsState.set_wandb_project_name,
            ),
            input_field("W&B Entity Name", on_blur=SettingsState.set_wandb_entity_name),
            input_field(
                "W&B API Key", type="password", on_blur=SettingsState.set_wandb_api_key
            ),
            input_field(
                "OpenAI API Key",
                type="password",
                on_blur=SettingsState.set_openai_api_key,
            ),
            rx.button(
                "Save",
                size="3",
                width="100%",
                on_click=SettingsState.authenticate,
            ),
            rx.foreach(
                SettingsState.authentication_messages,
                lambda text: error_message(
                    text,
                    color=rx.cond(text == "Authentication successful", "green", "red"),
                ),
            ),
            spacing="6",
            width="100%",
        ),
        size="4",
        max_width="28em",
        width="100%",
    )

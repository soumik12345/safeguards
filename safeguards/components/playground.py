import reflex as rx

from ..state.playground import PlaygroundState


def playground_header(text: str):
    return rx.hstack(
        rx.heading(text),
        align="center",
        width="100%",
        border_bottom=f"1.5px solid {rx.color('gray', 5, True)}",
        padding="1em",
    )


def select_playground_option():
    return rx.vstack(
        rx.hstack(
            rx.icon("bot-message-square"),
            rx.text("Chat Model"),
            align="center",
            width="100%",
            spacing="2",
        ),
        rx.box(
            rx.select(
                [
                    "gpt-4o",
                    "gpt-4o-mini",
                ],
                width="100%",
                value=PlaygroundState.playground_llm,
                on_change=PlaygroundState.set_playground_llm,
            ),
            width="100%",
        ),
        spacing="2",
        width="100%",
        padding="1em",
    )


def render_checkboxes(values, handler):
    return rx.vstack(
        rx.foreach(
            values,
            lambda choice: rx.checkbox(
                choice[0],
                checked=choice[1],
                on_change=lambda val: handler(val, choice[0]),
            ),
        )
    )


def select_guardrails():
    return rx.vstack(
        rx.hstack(
            rx.icon("shield"),
            rx.text("Guardrails"),
            align="center",
            width="100%",
            spacing="2",
        ),
        render_checkboxes(
            PlaygroundState.guardrail_choices,
            PlaygroundState.check_guardrail_choice,
        ),
        rx.text("Your choices: ", PlaygroundState.checked_guardrails),
        spacing="2",
        width="100%",
        padding="1em",
    )


def playground_ui():
    return rx.flex(
        rx.box(
            rx.vstack(
                playground_header("Playground Options"),
                select_playground_option(),
                select_guardrails(),
                spacing="2",
            ),
            flex=1,
            display="flex",
            flex_direction="column",
            border="1px solid #ccc",
        ),
        rx.box(
            playground_header("Playground Output"),
            flex=1,
            display="flex",
            flex_direction="column",
            border="1px solid #ccc",
        ),
        height="100vh",
    )

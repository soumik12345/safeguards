import reflex as rx


def navbar_link(text: str, url: str) -> rx.Component:
    return rx.link(rx.text(text, size="4", weight="medium"), href=url)


def navbar() -> rx.Component:
    logged_in_user_menu = rx.menu.root(
        rx.menu.trigger(
            rx.icon_button(
                rx.icon("user"),
                size="2",
                radius="full",
            )
        ),
        rx.menu.content(
            rx.menu.item("Settings", on_click=lambda: rx.redirect("/settings")),
        ),
        justify="end",
    )
    return rx.box(
        rx.desktop_only(
            rx.hstack(
                rx.link(
                    rx.heading("Safeguards", size="7", weight="bold"),
                    href="/",
                ),
                rx.hstack(
                    navbar_link("Playground", "/playground"),
                    navbar_link("Evaluations", "/#"),
                    spacing="5",
                ),
                logged_in_user_menu,
                justify="between",
                align_items="center",
            ),
        ),
        bg=rx.color("accent", 3),
        padding="1em",
        width="100%",
    )

import reflex as rx


class AppState(rx.State):
    openai_api_key: str = ""
    wandb_api_key: str = ""

import os
from typing import List

import reflex as rx
import rich
import wandb


class SettingsState(rx.State):
    wandb_project_name: str = ""
    wandb_entity_name: str = ""
    openai_api_key: str = ""
    wandb_api_key: str = ""
    authentication_messages: List[str] = []
    authentication_successful: bool = False

    @rx.event
    def debug_log(self):
        rich.print(f"{self.wandb_project_name=}")
        rich.print(f"{self.wandb_entity_name=}")
        rich.print(f"{self.openai_api_key=}")
        rich.print(f"{self.wandb_api_key=}")

    @rx.event
    def authenticate(self):
        self.authentication_messages = []
        if not self.wandb_project_name:
            self.authentication_messages.append("W&B Project Name is required")
            rich.print("W&B Project Name is required")
        if not self.wandb_entity_name:
            self.authentication_messages.append("W&B Entity Name is required")
            rich.print("W&B Entity Name is required")
        if not self.wandb_api_key:
            self.authentication_messages.append("W&B API Key is required")
            rich.print("W&B API Key is required")
        if not self.openai_api_key:
            self.authentication_messages.append("OpenAI API Key is required")
            rich.print("OpenAI API Key is required")
        else:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            try:
                wandb_login = wandb.login(
                    key=self.wandb_api_key, relogin=True, verify=True
                )
                if not wandb_login:
                    self.authentication_messages.append("W&B API Key is invalid")
                    rich.print("W&B API Key is invalid")
                else:
                    self.authentication_successful = True
                    self.authentication_messages.append("Authentication successful")
                    rich.print("Authentication successful")
            except Exception as e:
                self.authentication_messages.append(
                    f"Error during authentication: {str(e)}"
                )
                rich.print(f"Error during authentication: {str(e)}")

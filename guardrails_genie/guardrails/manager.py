import weave
from pydantic import BaseModel
from rich.progress import track

from .base import Guardrail


class GuardrailManager(weave.Model):
    """
    GuardrailManager is responsible for managing and executing a series of guardrails
    on a given prompt. It utilizes the `weave` framework to define operations that
    can be applied to the guardrails.

    Attributes:
        guardrails (list[Guardrail]): A list of Guardrail objects that define the
            rules and checks to be applied to the input prompt.
    """

    guardrails: list[Guardrail]

    @weave.op()
    def guard(self, prompt: str, progress_bar: bool = True, **kwargs) -> dict:
        """
        Execute a series of guardrails on a given prompt and return the results.

        This method iterates over a list of Guardrail objects, applying each guardrail's
        `guard` method to the provided prompt. It collects responses from each guardrail
        and compiles them into a summary report. The function also determines the overall
        safety of the prompt based on the responses from the guardrails.

        Args:
            prompt (str): The input prompt to be evaluated by the guardrails.
            progress_bar (bool, optional): If True, displays a progress bar while
                processing the guardrails. Defaults to True.
            **kwargs: Additional keyword arguments to be passed to each guardrail's
                `guard` method.

        Returns:
            dict: A dictionary containing:
                - "safe" (bool): Indicates whether the prompt is considered safe
                  based on the guardrails' evaluations.
                - "alerts" (list): A list of dictionaries, each containing the name
                  of the guardrail and its response.
                - "summary" (str): A formatted string summarizing the results of
                  each guardrail's evaluation.
        """
        alerts, summaries, safe = [], "", True
        iterable = (
            track(self.guardrails, description="Running guardrails")
            if progress_bar
            else self.guardrails
        )
        for guardrail in iterable:
            response = guardrail.guard(prompt, **kwargs)
            alerts.append(
                {"guardrail_name": guardrail.__class__.__name__, "response": response}
            )
            if isinstance(response, BaseModel):
                safe = safe and response.safe
                summaries += f"**{guardrail.__class__.__name__}**: {response.explanation}\n\n---\n\n"
            else:
                safe = safe and response["safe"]
                summaries += f"**{guardrail.__class__.__name__}**: {response['summary']}\n\n---\n\n"
        return {"safe": safe, "alerts": alerts, "summary": summaries}

    @weave.op()
    def predict(self, prompt: str, **kwargs) -> dict:
        """
        Predicts the safety and potential issues of a given input prompt using the guardrails.

        This function serves as a wrapper around the `guard` method, providing a simplified
        interface for evaluating the input prompt without displaying a progress bar. It
        applies a series of guardrails to the prompt and returns a detailed assessment.

        Args:
            prompt (str): The input prompt to be evaluated by the guardrails.
            **kwargs: Additional keyword arguments to be passed to each guardrail's
                `guard` method.

        Returns:
            dict: A dictionary containing:
                - "safe" (bool): Indicates whether the prompt is considered safe
                  based on the guardrails' evaluations.
                - "alerts" (list): A list of dictionaries, each containing the name
                  of the guardrail and its response.
                - "summary" (str): A formatted string summarizing the results of
                  each guardrail's evaluation.
        """
        return self.guard(prompt, progress_bar=False, **kwargs)

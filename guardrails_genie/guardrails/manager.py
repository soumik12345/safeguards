import weave
from rich.progress import track
from weave.flow.obj import Object as WeaveObject

from .base import Guardrail


class GuardrailManager(WeaveObject):
    guardrails: list[Guardrail]

    @weave.op()
    def guard(self, prompt: str, **kwargs) -> dict:
        alerts, safe = [], True
        for guardrail in track(self.guardrails, description="Running guardrails"):
            response = guardrail.guard(prompt, **kwargs)
            alerts.append(
                {"guardrail_name": guardrail.__class__.__name__, "response": response}
            )
            safe = safe and response["safe"]
        return {"safe": safe, "alerts": alerts}

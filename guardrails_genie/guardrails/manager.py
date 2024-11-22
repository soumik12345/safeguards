import weave
from weave.flow.obj import Object as WeaveObject

from .base import Guardrail


class GuardrailManager(WeaveObject):
    guardrails: list[Guardrail]

    @weave.op()
    def guard(self, prompt: str, **kwargs) -> dict:
        alerts = []
        for guardrail in self.guardrails:
            response = guardrail.guard(prompt, **kwargs)
            alerts.append({guardrail.name: response})
        return alerts

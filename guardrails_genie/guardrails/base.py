from abc import abstractmethod

import weave


class Guardrail(weave.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    @weave.op()
    def guard(self, prompt: str, **kwargs) -> list[str]:
        pass

    @weave.op()
    def predict(self, prompt: str, **kwargs) -> list[str]:
        return self.guard(prompt, **kwargs)

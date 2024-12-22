import reflex as rx


class PlaygroundState(rx.State):
    playground_llm: str = ""
    guardrail_choices: dict[str, bool] = {
        k: False
        for k in [
            "Prompt Injection: LLM Guardrail",
            "Prompt Injection: Classifier Guardrail",
        ]
    }

    def check_guardrail_choice(self, value, index):
        self.guardrail_choices[index] = value

    @rx.var
    def checked_guardrails(self):
        choices = [label for label, value in self.guardrail_choices.items() if value]
        return " / ".join(choices) if choices else "None"

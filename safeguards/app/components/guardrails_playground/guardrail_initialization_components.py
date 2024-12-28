from fasthtml.common import Div, Input, P


def PromptInjectionLLMGuardrailInitialization():
    return Div(
        Input(type="radio", name="my-accordion-3"),
        Div(
            "PromptInjectionLLMGuardrail: Advance Settings",
            cls="collapse-title text-xl font-medium",
        ),
        Div(P("PromptInjectionLLMGuardrail"), cls="collapse-content"),
        cls="collapse collapse-plus bg-base-200",
    )


def PromptInjectionClassifierGuardrailInitialization():
    return Div(
        Input(type="radio", name="my-accordion-3"),
        Div(
            "PromptInjectionClassifierGuardrail: Advance Settings",
            cls="collapse-title text-xl font-medium",
        ),
        Div(
            P("PromptInjectionClassifierGuardrail"),
            cls="collapse-content",
        ),
        cls="collapse collapse-plus bg-base-200",
    )

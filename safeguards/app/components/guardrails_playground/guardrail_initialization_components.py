from fasthtml.common import Div, Form, Input, Option, Select, Span


def PromptInjectionLLMGuardrailInitialization():
    return Div(
        Input(type="radio", name="my-accordion-3"),
        Div(
            "PromptInjectionLLMGuardrail: Advance Settings",
            cls="collapse-title text-xl font-medium",
        ),
        Div(
            Div(
                Form(
                    Div(
                        Span(
                            "Select LLM for PromptInjectionLLMGuardrail",
                            cls="label-text",
                        ),
                        cls="label",
                    ),
                    Select(
                        Option(""),
                        Option("gpt-4o"),
                        Option("gpt-4o-mini"),
                        cls="select select-bordered",
                        name="PromptInjectionLLMGuardrail_llm_model",
                    ),
                    cls="form-control w-full max-w-xs",
                ),
                cls="container mx-auto p-6",
            ),
            cls="collapse-content",
        ),
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
            Div(
                Form(
                    Div(
                        Span(
                            "Select LLM for PromptInjectionLLMGuardrail",
                            cls="label-text",
                        ),
                        cls="label",
                    ),
                    Input(
                        type="text",
                        cls="input input-bordered",
                        name="PromptInjectionClassifierGuardrail_model_name",
                    ),
                    cls="form-control w-full max-w-xs",
                ),
                cls="container mx-auto p-6",
            ),
            cls="collapse-content",
        ),
        cls="collapse collapse-plus bg-base-200",
    )

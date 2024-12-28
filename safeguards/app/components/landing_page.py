from fasthtml.components import H1, Button, Div, P


def SafeGuardsLanding():
    return Div(
        Div(
            Div(
                H1("SafeGuards", cls="text-5xl font-bold"),
                P(
                    "SafeGuards is a platform for implementing guardrails and safety checks for your LLM applications.",
                    cls="py-6",
                ),
                Button("Get Started", cls="btn btn-primary"),
                cls="max-w-md",
            ),
            cls="hero-content text-center",
        ),
        cls="hero bg-base-200 min-h-screen",
    )

from fasthtml.components import A, Details, Div, Li, Span, Summary, Ul


def SafeGuardsNavBar():
    return Div(
        Div(A("SafeGuards", cls="btn btn-ghost text-xl"), cls="flex-1"),
        Div(
            Ul(
                Li(
                    Details(
                        Summary("GuardRails"),
                        Ul(
                            Li(A("Playground", href="/guardrails_playground")),
                            Li(A("Evaluation", href="#")),
                            cls="bg-base-100 rounded-t-none p-2",
                        ),
                    )
                ),
                Li(A("Settings", href="/settings")),
                cls="menu menu-horizontal px-1",
            ),
            cls="flex-none",
        ),
        cls="navbar bg-base-100",
    )


def StatusModal(message: str, success: bool):
    return Div(
        Div(
            Span(message), cls="alert alert-success" if success else "alert alert-error"
        ),
        cls="toast",
    )

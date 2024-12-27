from fasthtml.components import A, Details, Div, Li, Summary, Ul


def SafeGuardsNavBar():
    return Div(
        Div(A("SafeGuards", cls="btn btn-ghost text-xl"), cls="flex-1"),
        Div(
            Ul(
                Li(A("Any Chat", href="#")),
                Li(A("Any Imagine", href="#")),
                Li(
                    Details(
                        Summary("GuardRails"),
                        Ul(
                            Li(A("Playground", href="#")),
                            Li(A("Evaluation", href="#")),
                            cls="bg-base-100 rounded-t-none p-2",
                        ),
                    )
                ),
                Li(A("Settings", href="#")),
                cls="menu menu-horizontal px-1",
            ),
            cls="flex-none",
        ),
        cls="navbar bg-base-100",
    )

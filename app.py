from fasthtml.common import FastHTML, FileResponse, serve, Link, Button
from safeguards.app.tailwind import add_daisy_and_tailwind


app = FastHTML(
    hdrs=Link(rel="stylesheet", href="app.css", type="text/css")
)
app.static_route_exts(static_path="public")
add_daisy_and_tailwind(app)
route = app.route

@app.get("/{fname:path}.{ext:static}")
def get_static(fname: str, ext: str):
    return FileResponse(f"public/{fname}.{ext}")


@app.get("/")
def get_index():
    return Button("Button", cls="btn")


serve()
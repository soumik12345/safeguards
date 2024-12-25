from fasthtml.core import serve
from monsterui.core import FastHTML, Theme

from safeguards.app.navbar import SafeGuardsNavBar
from safeguards.app.settings import SettingsPage

app = FastHTML(hdrs=Theme.blue.headers())
route = app.route


@app.get("/")
def landing_page():
    return SafeGuardsNavBar()


@app.get("/settings")
def settings_page():
    return SettingsPage()


serve()

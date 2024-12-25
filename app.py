from fasthtml.core import serve
from monsterui.core import FastHTML, Theme

from safeguards.app.navbar import SafeGuardsNavBar

app = FastHTML(hdrs=Theme.blue.headers())
route = app.route


@route("/")
def get():
    return SafeGuardsNavBar()


serve()

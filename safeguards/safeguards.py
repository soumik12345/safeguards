import reflex as rx

from .pages import index, playground, settings

app = rx.App()
app.add_page(index)
app.add_page(settings)
app.add_page(playground)
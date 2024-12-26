from monsterui.core import P
from monsterui.daisy import Alert, AlertT
from monsterui.franken import DivLAligned, UkIcon


def AlertStatusNotification(message: str, success: bool):
    if not success:
        return Alert(
            DivLAligned(UkIcon("triangle-alert"), P(message)), cls=AlertT.error
        )
    return Alert(DivLAligned(UkIcon("check"), P(message)), cls=AlertT.success)

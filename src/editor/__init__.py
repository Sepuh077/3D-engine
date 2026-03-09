from PySide6 import QtWidgets
from .window import EditorWindow

def run_editor(project_root: str) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    editor = EditorWindow(project_root)
    editor.show()
    app.exec()

from PySide6 import QtWidgets
from .window import EditorWindow
from .undo import (
    UndoManager,
    Command,
    AddGameObjectCommand,
    DeleteGameObjectCommand,
    SelectObjectsCommand,
    AddComponentCommand,
    DeleteComponentCommand,
    FieldChangeCommand,
    RenameGameObjectCommand,
    ReparentGameObjectCommand,
    get_undo_manager,
    set_undo_manager,
)

def run_editor(project_root: str) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    editor = EditorWindow(project_root)
    editor.show()
    app.exec()

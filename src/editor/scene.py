from src.engine3d.scene import Scene3D

class EditorScene(Scene3D):
    def __init__(self) -> None:
        super().__init__()
        self.editor_label = "Untitled Scene"

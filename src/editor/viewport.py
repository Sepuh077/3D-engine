from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets, QtOpenGLWidgets

class ViewportWidget(QtOpenGLWidgets.QOpenGLWidget):
    resized = QtCore.Signal(int, int)
    file_dropped = QtCore.Signal(str)
    
    # Mouse events for camera control
    mouse_pressed = QtCore.Signal(QtGui.QMouseEvent)
    mouse_released = QtCore.Signal(QtGui.QMouseEvent)
    mouse_moved = QtCore.Signal(QtGui.QMouseEvent)
    wheel_scrolled = QtCore.Signal(QtGui.QWheelEvent)

    # Keyboard events
    key_pressed = QtCore.Signal(QtGui.QKeyEvent)
    key_released = QtCore.Signal(QtGui.QKeyEvent)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        fmt = QtGui.QSurfaceFormat()
        fmt.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
        fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setVersion(3, 3)
        super().__init__(parent)
        self.setFormat(fmt)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)  # Track mouse without clicking
        self.setAcceptDrops(True)
        self.render_callback = None

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() or event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            event.acceptProposedAction()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path:
                    self.file_dropped.emit(path)
        elif event.mimeData().hasFormat("application/x-qabstractitemmodeldatalist"):
            # This handles drag from the internal QTreeView
            # But it's easier to just use the selection from the tree in the window
            # For now, we'll let the window handle the drop if it's from the tree
            # or just emit a special signal.
            # Actually, the window can just check the tree selection when a drop happens.
            self.file_dropped.emit("") # Signal that something was dropped from the tree
        
        event.acceptProposedAction()

    def paintGL(self) -> None:
        if self.render_callback:
            self.render_callback()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        size = event.size()
        dpr = self.devicePixelRatio()
        self.resized.emit(int(size.width() * dpr), int(size.height() * dpr))
    
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.mouse_pressed.emit(event)
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self.mouse_released.emit(event)
        super().mouseReleaseEvent(event)
    
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self.mouse_moved.emit(event)
        super().mouseMoveEvent(event)
    
    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        self.wheel_scrolled.emit(event)
        super().wheelEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        self.key_pressed.emit(event)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        self.key_released.emit(event)
        super().keyReleaseEvent(event)

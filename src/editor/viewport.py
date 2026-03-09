from typing import Optional
from PySide6 import QtCore, QtGui, QtWidgets, QtOpenGLWidgets

class ViewportWidget(QtOpenGLWidgets.QOpenGLWidget):
    resized = QtCore.Signal(int, int)
    
    # Mouse events for camera control
    mouse_pressed = QtCore.Signal(QtGui.QMouseEvent)
    mouse_released = QtCore.Signal(QtGui.QMouseEvent)
    mouse_moved = QtCore.Signal(QtGui.QMouseEvent)
    wheel_scrolled = QtCore.Signal(QtGui.QWheelEvent)

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
        self.render_callback = None

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
